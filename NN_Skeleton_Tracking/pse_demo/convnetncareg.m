function [f, df] = convnetncareg(VV,Dim,x,y, ...
  connections,weightdecay,useabsrec)

  % The main function for the convnet (NCA Regression objective)
  % It first does a forward pass through the convnet and projects down to
  % the low-dimensional space
  % It then computes the NCA Regression cost, f
  % 
  % Then it performs backpropagation through the convnet and returns the
  % gradient of the cost, f, with respect to every parameter
  %
  % INPUTS:
  % VV : parameters flattened into a vector
  % Dim : a vector of various dimensions
  % x : images (represented as vectors)
  % y : pose labels
  % connections : connection table between 1st output map and 2nd input
  % map in convnet
  % weightdecay : regularization amount
  % useabsrec : binary variable indicating whether to use absolute value
  % rectification after filtering + nonlinearity
  
[numpixels,numcases] = size(x);
nr=sqrt(numpixels);
assert(nr==floor(sqrt(numpixels))) %assert square
nc=nr;

num_connect = size(connections,2);

filtersize1=Dim(1);
nummaps1=Dim(2);
downsample1=Dim(3);
filtersize2=Dim(4);
nummaps2=Dim(5);
downsample2=Dim(6);
outsize=Dim(7);
numhid=Dim(8);

outr1=nr-filtersize1+1;
outc1=nc-filtersize1+1;
nr1 = outr1/downsample1; nc1 = outc1/downsample1; %downsampled dimensions

outr2=nr1-filtersize2+1;
outc2=nc1-filtersize2+1;
nr2 = outr2/downsample2; nc2 = outc2/downsample2;

%deconversion of vectorized parameters
filters1 = reshape(VV(1:filtersize1*filtersize1*nummaps1),[filtersize1*filtersize1 nummaps1]);
xxx = filtersize1*filtersize1*nummaps1;
b1 = reshape(VV(xxx+1:xxx+nummaps1),1,nummaps1);
xxx = xxx+nummaps1;
convcoeff1 = reshape(VV(xxx+1:xxx+nummaps1),nummaps1,1);
xxx = xxx+nummaps1;
filters2 = reshape(VV(xxx+1:xxx+filtersize2*filtersize2*(num_connect*nummaps2)),[filtersize2*filtersize2 num_connect*nummaps2]);
xxx = xxx+filtersize2*filtersize2*(num_connect*nummaps2);
b2 = reshape(VV(xxx+1:xxx+nummaps2),1,nummaps2);
xxx=xxx+nummaps2;
convcoeff2 = reshape(VV(xxx+1:xxx+nummaps2),nummaps2,1);
xxx = xxx+nummaps2;
A = reshape(VV(xxx+1:xxx+numhid*outsize),numhid,outsize);

% Copy data and parameters to device memory
x_d=GPUsingle(x);
y_d=GPUsingle(y);

filters1_d=GPUsingle(filters1);
b1_d=GPUsingle(b1);
convcoeff1_d=GPUsingle(convcoeff1);
filters2_d=GPUsingle(filters2);
b2_d=GPUsingle(b2);
convcoeff2_d=GPUsingle(convcoeff2);
A_d=GPUsingle(A);
%w_class_d=GPUsingle(w_class);

%forward pass
%returns output map of top convolution layer (used in backprop)
[yy,sigmap1,map1,y1,sigmap2,map2] = convnet_forward2_gpu(x_d,filters1_d,b1_d, ...
  convcoeff1_d,downsample1,filters2_d, ...
    b2_d,convcoeff2_d,downsample2,connections,useabsrec);

  z = A_d*yy; %low-dim code in which we will find neighbours

  d = zeros(numcases,numcases,GPUsingle);
  cuSquaredDist(z,d);
  d1 = double(d); %copy to CPU
    
  d_y = zeros(numcases,numcases,GPUsingle);
  cuSquaredDist(y_d,d_y);
  d_y1 = double(d_y); %to CPU

  %Say all distances are very large
  %Then we could get exp(-d) all underflowing to zero
  %And denominator could be zero
  %To prevent this, subtract the min (not counting the diagonal/identity) of
  %each column
  %This is done on CPU since we don't have a GPUmax
  d1(1:numcases+1:end)=Inf; %otherwise identity will be the min
  [nv,nn] = min(d1,[],1); %find min distance in each column
  d1 = bsxfun(@minus,d1,nv); %subtract min 
  d1 = exp(-d1);
  d1(1:numcases+1:end)=0; %0 prob on diagonal
  denom = sum(d1,1);
  p = bsxfun(@rdivide,d1,denom);
  del = sum(p.*d_y1,1); %cost per cases
  f = sum(del,2); %total cost
  f = f/numcases + 0.5*weightdecay*( sum(filters1(:).^2) + sum(filters2(:).^2) + ...
    sum(A(:).^2)); %regularization

  

if nargout > 1 %then compute gradients  (on GPU)

  %d = GPUsingle(p); 
  memCpyHtoD(d,single(p),1,numel(p)); %expected below for gradients (p becomes d)
  del = GPUsingle(del); %expected below for gradients
  
  %compute gradient of the cost with respect to the low-dimensional rep
  %(i.e. the output of the last layer)

  %first, determine weights
  GPUminus(d_y,repmat(del,[numcases 1]),d_y);
  GPUtimes(d,d_y,d);
  GPUplus(d,d',d);

  dfdz = zeros(numcases,numhid,GPUsingle); %note GPU computes transpose
                                           %of what we want
  cuNCAreg(z,d,dfdz);
  dfdz = transpose(dfdz); 
  clear z d
  
  %gradient of f with respect to the input of the code layer
  dfdz = -(2/numcases)*dfdz;
  dfdA = dfdz*yy';
  dfdA = dfdA + weightdecay*A_d;
  delta4=A_d'*dfdz;  
  clear yy dfdz
  
  %subsampling layer has no parameters
  %if the layer following the subsampling layer is a fully connected layer
  %then the sensitivity maps can be computed by vanilla backprop
  %delta4 is in vector form
  %but now we need to reshape it so we can do the element wise multiplication
  %This is simply reversing the reshaping operations performed in the forward
  %pass
  
  setSize(delta4,[nr2*nc2,nummaps2*numcases]);

  delta3 = zeros(outr2*outc2,numcases*nummaps2,GPUsingle);
  cuSupersample(delta4,delta3,downsample2);
  clear delta4
  GPUtimes(delta3,(1/(downsample2*downsample2)),delta3); %due to averaging
  
  dconvcoeff2 = delta3.*map2;
  clear map2
  dconvcoeff2 = sum(dconvcoeff2,1); %sum over pixels
  setSize(dconvcoeff2,[nummaps2 numcases]);
  
  dconvcoeff2 = sum(dconvcoeff2,2); %sum over cases
    
  GPUtimes(delta3,repmat(convcoeff2_d',[outr2*outc2,numcases]),delta3);
  if useabsrec
    s = -(sigmap2<0)+(sigmap2>0);%sign(sigmap1)
    GPUtimes(delta3,s,delta3);
  end
  GPUtimes(delta3,1-sigmap2.^2,delta3); %tanh nonlinearity
  clear sigmap2
  
  setSize(delta3,[outr2*outc2 nummaps2 numcases]);
  db2=sum(delta3,3);
  db2=sum(db2,1);

  setSize(delta3,[outr2*outc2 nummaps2 numcases]);
  
  dfilters2 = zeros(filtersize2*filtersize2,num_connect*nummaps2,GPUsingle);
  outDims=filtersize2*filtersize2;
  for ii=1:nummaps1
    %linear index to each time input jj appears in connections
    input_map_list = find(connections==ii);    
    %ridx gives the output map
    %cidx gives the filter # for this output map
    if ~isempty(input_map_list)
      [ridx,cidx]=ind2sub([nummaps2 num_connect],input_map_list);      
      numFilters = length(ridx); 
      rnumFilters= 16*div((numFilters+16-1),16); %round up to multiple of 16
      filterBank = zeros(outr2*outc2*rnumFilters,numcases,GPUsingle);
      filterBank(1:outr2*outc2*numFilters,:)= ...
        reshape(delta3(:,ridx,:),[outr2*outc2*numFilters numcases]);
      convResult = zeros(outDims*rnumFilters,numcases,GPUsingle);
      inputMaps = reshape(y1(:,:,ii,:),[nr1*nc1 numcases]);
      cuConv2(inputMaps,filterBank,convResult,outr2);
      filteridx=num_connect*(ridx-1)+cidx; %identify which filters are used
      dfilters2(:,filteridx) = ...
        reshape(sum(convResult(1:outDims*numFilters,:),2),outDims,numFilters);
    end
  end
  cuRotate180(dfilters2,dfilters2);
  dfilters2=dfilters2+weightdecay*filters2_d;
    
  rnumFilters= 16*div((num_connect+16-1),16); %round up to multiple of 16
  bigInputMaps = zeros((outr2+2*(filtersize2-1))*(outc2+2*(filtersize2-1)),numcases,GPUsingle);
  delta2 = zeros(nr1*nc1,nummaps1,numcases,GPUsingle);
  outPixels=nr1*nc1;
  for jj=1:nummaps2 %loop through each output map
    input_map_list = connections(jj,:); %input maps which are connected to this output
    filteridx=num_connect*(jj-1)+1:(num_connect)*jj; %identify filters
    filterBank = zeros(filtersize2*filtersize2,rnumFilters,GPUsingle);
    filterBank(:,1:num_connect)=filters2_d(:,filteridx);
    convResult = zeros(outPixels*rnumFilters,numcases,GPUsingle);
    inputMaps = delta3(:,jj,:); %just this output map
    setSize(inputMaps,[outr2*outc2 numcases]);
  
    %full convolution instead of 'valid' (so we need padding)
    cuCopyInto(inputMaps,bigInputMaps,filtersize2-1); %zero padding
    cuConv(bigInputMaps,filterBank,convResult);
    %add these contributions to the appropriate input delta
    for ii=1:num_connect
      delta2(:,input_map_list(ii),:) = delta2(:,input_map_list(ii),:) + ...
        reshape(convResult(outPixels*(ii-1)+1:outPixels*ii,:),[outPixels 1 numcases]);
    end
  end
  clear delta3 convResult inputMaps bigInputMaps
  
  setSize(delta2,[nr1*nc1,nummaps1*numcases]);
  
  delta1 = zeros(outr1*outc1,numcases*nummaps1,GPUsingle);
  cuSupersample(delta2,delta1,downsample1);
  clear delta2
  
  GPUtimes(delta1,(1/(downsample1*downsample1)),delta1); %due to averaging
  
  dconvcoeff1 = delta1.*map1;
  clear map1
  dconvcoeff1 = sum(dconvcoeff1,1); %sum over pixels

  setSize(dconvcoeff1,[nummaps1 numcases]);
  dconvcoeff1 = sum(dconvcoeff1,2); %sum over cases
  
  GPUtimes(delta1,repmat(convcoeff1_d',[outr1*outc1,numcases]),delta1);
  if useabsrec
    s = -(sigmap1<0);
    s1 = (sigmap1>0);
    GPUplus(s,s1,s); %sign(sigmap1)

    GPUtimes(delta1,s,delta1);
  end
  GPUpower(sigmap1,2,sigmap1);
  GPUminus(1,sigmap1,sigmap1);
  GPUtimes(delta1,sigmap1,delta1);

  clear sigmap1
  
  setSize(delta1,[outr1*outc1 nummaps1 numcases]);
  db1=sum(delta1,3);
  db1=sum(db1,1);

  setSize(delta1,[outr1*outc1*nummaps1,numcases]);

  dfilters1 = zeros(nummaps1*filtersize1*filtersize1,numcases,GPUsingle);
  cuConv2(x_d,delta1,dfilters1,outr1); %filterSize is the size of deltas!
  clear delta1 x_d
  dfilters1=sum(dfilters1,2); %sum over cases

  setSize(dfilters1,[filtersize1*filtersize1 nummaps1]);
  cuRotate180(dfilters1,dfilters1);
  dfilters1=dfilters1+weightdecay*filters1_d;

  df = [dfilters1(:);db1(:);dconvcoeff1(:);dfilters2(:);db2(:);dconvcoeff2(:);dfdA(:)];
  df = double(df);
end
