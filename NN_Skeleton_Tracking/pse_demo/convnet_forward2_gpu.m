function [y2,resp1,map1,y1,resp2,map2] = convnet_forward2_gpu(data,filters1,b1,convcoeff1, ...
                                      downsample1,filters2,b2,convcoeff2, ...
                                      downsample2,connections,useabsrec)

%takes a mini-batch of data and does a forward pass through the
%convolutional net
%data and filters should be GPUsingle types
%downsample : downsampling factor
%returns: y2: outputs of final subsampling layer in vector form
%         resp1: output of nonlinearity at 1st convolution layer
%         map1: output of rectification at 1st convolution layer (if useabsrec=1)
%         y1:   output of 1st subsampling layer
%         resp1: output of nonlinearity at 2nd convolution layer
%         map1: output of rectification at 2nd convolution layer (if useabsrec=1)
%
% INPUTS: 
% data : image data
% filters1 : 1st layer filters
% b1 : 1st layer additive bias
% convcoeff1 : 1st layer gain (applied after nonlinearity)
% downsample1 : 1st layer downsampling factor
% filters2 : 2nd layer filters
% b2 : 2nd layer additive bias
% convcoeff2 : 2nd layer gain (applied after nonlinearity)
% downsample2 : 2nd layer downsampling factor

[numpixels,numcases]=size(data);
nr=sqrt(numpixels);
assert(nr==floor(sqrt(numpixels))) %assert square
nc=nr;

[numfilterpixels1,nummaps1] = size(filters1);
filtersize1=sqrt(numfilterpixels1);
assert(filtersize1==floor(sqrt(numfilterpixels1))) %assert square

outr1=nr-filtersize1+1;
outc1=nc-filtersize1+1;
nr1 = outr1/downsample1; nc1 = outc1/downsample1; %downsampled dimensions

num_connect = size(connections,2);
nummaps2 = size(filters2,2)/num_connect;

numfilterpixels2 = size(filters2,1);
filtersize2=sqrt(numfilterpixels2);
assert(filtersize2==floor(sqrt(numfilterpixels2))) %assert square

outr2=nr1-filtersize2+1;
outc2=nc1-filtersize2+1;
nr2 = outr2/downsample2; nc2 = outc2/downsample2;

resp1 = zeros(nummaps1*outr1*outc1,numcases,GPUsingle); %filter responses

resp2 = zeros(outr2*outc2,nummaps2,numcases,GPUsingle); %filter responses


%we don't want to rotate filters1 in place
%since they are passed by reference and used again in calling function
filters1_r=clone(filters1);
cuRotate180(filters1,filters1_r); % since cuConv actually correlates
                                  % rather than convolves
cuConv(data,filters1_r,resp1); % convolve filters with input

%nonlinearity
setSize(resp1,[outr1*outc1,nummaps1*numcases]);
GPUplus(resp1,repmat(b1,[outr1*outc1,numcases]),resp1); %add bias
GPUtanh(resp1,resp1);

if useabsrec
  map1=abs(resp1);
else
  map1=resp1;
end

y1 = zeros(nr1*nc1,nummaps1*numcases,GPUsingle);
cuSubsample(map1,y1,downsample1);
GPUtimes(y1,repmat(convcoeff1',[nr1*nc1,numcases]),y1); %apply coefficient
y1 = reshape(y1,[nr1 nc1 nummaps1 numcases]);

% Convolution with 2nd layer filters is more involved since it depends on
% the connectivity map
% cuConv currently has the limitation that the # of filters needs to be a
% multiple of 16
outPixels = outr2*outc2;
for ii=1:nummaps1 %loop through each input map
  %linear index to each time input jj appears in connections
  input_map_list = find(connections==ii);
  %ridx gives the output map
  %cidx gives the filter # for this output map
  if ~isempty(input_map_list)
    [ridx,cidx]=ind2sub([nummaps2 num_connect],input_map_list);
    numFilters = length(ridx); 
    rnumFilters= 16*div((numFilters+16-1),16); %round up to multiple of 16
    filterBank = zeros(filtersize2*filtersize2,rnumFilters,GPUsingle);
    filteridx=num_connect*(ridx-1)+cidx; %identify which filters are used
    filterBank(:,1:numFilters)=filters2(:,filteridx);
    cuRotate180(filterBank,filterBank);
    convResult = zeros(outPixels*rnumFilters,numcases,GPUsingle);
    inputMaps = reshape(y1(:,:,ii,:),[nr1*nc1 numcases]);
    cuConv(inputMaps,filterBank,convResult);
    %now we need to add these contributions to the appropriate output
    for jj=1:numFilters %can we do this without a loop?
      resp2(:,ridx(jj),:)=resp2(:,ridx(jj),:) + ...
        reshape(convResult(outPixels*(jj-1)+1:outPixels*jj,:),[outPixels 1 numcases]);
    end
  end  
end

setSize(resp2,[outr2*outc2,nummaps2*numcases]);
GPUplus(resp2,repmat(b2,[outr2*outc2,numcases]),resp2); %add bias

%nonlinearity
GPUtanh(resp2,resp2);

if useabsrec
  map2=abs(resp2);
else
  map2=resp2;
end

y2 = zeros(nr2*nc2,nummaps2*numcases,GPUsingle);
cuSubsample(map2,y2,downsample2);
GPUtimes(y2,repmat(convcoeff2',[nr2*nc2,numcases]),y2); %apply coefficient
setSize(y2,[nr2 nc2 nummaps2 numcases]);

%now we feed the output (i.e. all maps) to a classifier
%basically we want to flatten here (except for cases)
setSize(y2,[nr2*nc2*nummaps2,numcases]); %flatten