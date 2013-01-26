function [err_train,err_valid,err_validmax,err_pix] = ...
  ncaregeval_gpu(VV,Dim,traindata,trainlabels,validdata,validlabels,connections,useabsrec,skiphead);

  %Compute various error metrics
  %err_train NCA regression cost (training data)
  %err_valid NCA regression cost (validation data)
  %err_validmax NCA regression cost (ignoring all but 1NN)
  %err_pix Pixel error using the 1NN
  
if nargin<9
  skiphead=1; %don't count head when evaluating pixel error
end

batchsize=128; %batch size in which to do up-pass

[numpixels,numcases] = size(traindata);
nr=sqrt(numpixels);
assert(nr==floor(sqrt(numpixels))) %assert square
nc=nr;
[numpixels1,numcasesvalid] = size(validdata);

num_connect = size(connections,2);

assert(numpixels==numpixels1)

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

%GPU types
%target_d=GPUsingle(target);
filters1_d=GPUsingle(filters1);
b1_d=GPUsingle(b1);
convcoeff1_d=GPUsingle(convcoeff1);
filters2_d=GPUsingle(filters2);
b2_d=GPUsingle(b2);
convcoeff2_d=GPUsingle(convcoeff2);
A_d=GPUsingle(A);

%data to GPU
%traindata_d = GPUsingle(traindata);
trainlabels_d = GPUsingle(trainlabels);

%The up-pass is GPU memory intensive
%So we may need to batch this up
z = zeros(numhid,numcases,GPUsingle); %to hold codes
numbatches=div((numcases+batchsize-1),batchsize); %round up
for bb=1:numbatches
  batchidx=(bb-1)*batchsize+1:min(bb*batchsize,numcases);
  data_d = GPUsingle(traindata(:,batchidx)); %avoids putting all data into GPU mem
  [y2,~,~,~,~,~] = convnet_forward2_gpu(data_d,filters1_d,b1_d, ...
    convcoeff1_d,downsample1,filters2_d, ...
    b2_d,convcoeff2_d,downsample2,connections,useabsrec);
  z(:,batchidx)=A_d*y2;
  %yy(:,batchidx)=y2;
end
clear data_d y2
%z = A_d*yy; %code in which we will find neighbours

if numcases>10000 %large numcases not handled yet
  f=0;
else
  %distance matrix in z space
  %d = distmatbsxfun_t(z); %takes data as cols
  d = zeros(numcases,numcases,GPUsingle);
  cuSquaredDist(z,d);
  d = double(d); %move back to CPU
  
  %distance matrix in y (label) space
  % d_y = distmatbsxfun_t(labels); %takes data as cols
  d_y = zeros(numcases,numcases,GPUsingle);
  cuSquaredDist(trainlabels_d,d_y);
  d_y = double(d_y); %back to CPU
  
  %Say all distances are very large
  %Then we could get exp(-d) all underflowing to zero
  %And denominator could be zero
  %To prevent this, subtract the min (not counting the diagonal/identity) of
  %each column
  d(1:numcases+1:end)=Inf; %otherwise identity will be the min
  [nv,nn] = min(d,[],1); %find min distance in each column
  d = bsxfun(@minus,d,nv); %subtract min
  d = exp(-d);
  d(1:numcases+1:end)=0; %0 prob on diagonal
  denom = sum(d,1);
  p = bsxfun(@rdivide,d,denom);
  
  del = sum(p.*d_y,1); %this is the cost per point
  
  f = sum(del,2); %sum over cases

end

err_train=sqrt(f/(numcases));

%validdata_d = GPUsingle(validdata);
validlabels_d = GPUsingle(validlabels);

%Now with validation set
zv = zeros(numhid,numcasesvalid,GPUsingle); %to hold codes
numbatches=div((numcasesvalid+batchsize-1),batchsize); %round up
for bb=1:numbatches
  batchidx=(bb-1)*batchsize+1:min(bb*batchsize,numcasesvalid);
  data_d = GPUsingle(validdata(:,batchidx)); %avoids putting all data into GPU mem
  [y2,~,~,~,~,~] = convnet_forward2_gpu(data_d,filters1_d,b1_d, ...
    convcoeff1_d,downsample1,filters2_d, ...
    b2_d,convcoeff2_d,downsample2,connections,useabsrec);
  zv(:,batchidx)=A_d*y2;
  %yy(:,batchidx)=y2;
end
clear data_d y2
%clear validdata_d
%zv = A_d*yy; %code in which we will find neighbours
%clear yy

if numcasesvalid>5000 %large numcasesvalid not handled yet
  f=0;
else
  d = zeros(numcases,numcasesvalid,GPUsingle);
  cuSquaredDist(zv,z,d);
  d = double(d); %back to CPU
  clear z zv
  
  %d_y = distmatbsxfun_t(labels,validlabels);
  d_y = zeros(numcases,numcasesvalid,GPUsingle);
  cuSquaredDist(validlabels_d,trainlabels_d,d_y);
  d_y = double(d_y); %back to CPU
  clear validlabels_d trainlabels_d
  
  [nv,nn] = min(d,[],1); %find min distance in each column
  d = bsxfun(@minus,d,nv); %subtract min
  d = exp(-d);
  %d(1:numcases+1:end)=0; %0 prob on diagonal
  denom = sum(d,1);
  p = bsxfun(@rdivide,d,denom);
  
  del = sum(p.*d_y,1); %this is the cost per point
  
  f = sum(del,2); %sum over cases
end

err_valid=sqrt(f/numcasesvalid);

if numcasesvalid>5000
  f=0;
  err_pix = 0;
else
  
  %Slightly different way to measure validation error
  %Only take the nearest neighbour and use its value
  [nv,nn] = max(p,[],1); %nn gives indices
  %create linear index
  idx = sub2ind([numcases numcasesvalid],nn,1:numcasesvalid);
  del = d_y(idx);
  
  f = sum(del,2); %sum over cases
  
  %Error in pixels
  err_pix = compute_pixel_error(nn,trainlabels,validlabels,skiphead); %on CPU

end

err_validmax=sqrt(f/numcasesvalid);

