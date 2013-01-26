function [err_train,err_trainmax,err_valid,err_validmax,err_pix] = ...
  convnetdrlimeval_gpu(VV,Dim,traindata,trainlabels,validdata,validlabels,m,connections,useabsrec,skiphead);

if nargin<10
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
  d1 = double(d); %copy to CPU
  
  %n = sqrt(d); %Euclidean norm
  GPUsqrt(d,d); %in-place sqrt
  n1 = double(d); %copy to CPU
  clear d
  
  d_y = zeros(numcases,numcases,GPUsingle);
  cuSquaredDist(trainlabels_d,d_y);
  d_y = double(d_y); %move to CPU
  
  %we need the real d_y later so that's why we make d_y1 here
  %now we want to normalize d_y
  d_y1 = d_y+diag(nan(1,numcases)); %NaN on diagonals
  [nv,nn] = min(d_y1,[],1); %find nearest neighbour
  d_y1 = bsxfun(@minus,d_y1,nv); %make sure the thing we subtract doesn't get squared

  negexpds = exp(-(d_y1)); %note that d has already been squared
  negexpds(1:numcases+1:end)=0; %0 prob on diagonal

  denom = sum(negexpds,1);

  p = bsxfun(@rdivide,negexpds,denom); %note that p is based on the pose labels
 
  %used a lot
  np = 1-p;
  
  %Similarity loss
  fs = 0.5*sum(sum(p.*d1));
  
  %Dissimilarity loss
  %Note that m is defined on norm, not squared norm
  md = bsxfun(@max,0,m-n1); %only retains distances under margin
  md(1:size(md,1)+1:end)=0; %diagonal elements should not contribute
  
  fd = 0.5*sum(sum(np.*md.^2));
  
  f = fs + fd; %similarity and dissimilarity loss
  
end

err_train=f/numcases;

if numcases>10000 %large numcases not handled yet
  f=0;
else
  %assume d1 has already been computed
  %What about simple nearest neighbour on the training set?
  %Where we don't allow matches between the same neighbour
  d1(1:numcases+1:end)=Inf;
  %Slightly different way to measure validation error
  %Only take the nearest neighbour and use its value
  [nv,nn] = min(d1,[],1); %nn gives indices
                          %create linear index
  idx = sub2ind([numcases numcases],nn,1:numcases);
  del = d_y(idx); %squared distance to next nearest neighbour (in pose space)

  f = sum(del,2); %sum over cases
  
end
err_trainmax=sqrt(f/numcases);

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
  d1 = double(d); %back to CPU
  
  %n = sqrt(d); %Euclidean norm
  GPUsqrt(d,d); %in-place sqrt
  n1 = double(d); %copy to CPU
  clear d
    
  clear z zv
  
  %d_y = distmatbsxfun_t(labels,validlabels);
  d_y = zeros(numcases,numcasesvalid,GPUsingle);
  cuSquaredDist(validlabels_d,trainlabels_d,d_y);
  d_y = double(d_y); %back to CPU
  clear validlabels_d trainlabels_d

  %we need d_y later, so we create d_y1 here
  %now we want to normalize d_y

  [nv,nn] = min(d_y,[],1); %find nearest neighbour
  d_y1 = bsxfun(@minus,d_y,nv); %make sure the thing we subtract doesn't get squared

  negexpds = exp(-(d_y1)); %note that d has already been squared
                           %negexpds(1:numcases+1:end)=0; %0 prob on diagonal

  denom = sum(negexpds,1);

  p = bsxfun(@rdivide,negexpds,denom); %note that p is based on the pose labels

  %used a lot
  np = 1-p;
  
  %Similarity loss
  fs = 0.5*sum(sum(p.*d1));
  
  %Dissimilarity loss
  %Note that m is defined on norm, not squared norm
  md = bsxfun(@max,0,m-n1); %only retains distances under margin

  fd = 0.5*sum(sum(np.*md.^2));
  
  f = fs + fd; %similarity and dissimilarity loss
  
end

err_valid=sqrt(f/numcasesvalid);

if numcasesvalid>5000
  f=0;
else
  
  %Slightly different way to measure validation error
  %Only take the nearest neighbour and use its value
  [nv,nn] = min(d1,[],1); %nn gives indices
  %create linear index
  idx = sub2ind([numcases numcasesvalid],nn,1:numcasesvalid);
  del = d_y(idx);
  
  f = sum(del,2); %sum over cases
end

err_validmax=sqrt(f/numcasesvalid);

%Error in pixels
err_pix = compute_pixel_error(nn,trainlabels,validlabels,skiphead); %on CPU
