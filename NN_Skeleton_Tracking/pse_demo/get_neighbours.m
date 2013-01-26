function [nn_idx,nn_v] = get_neighbours(VV,Dim,querypixels,databasepixels,connections,useabsrec,dg)

%Given a numpixelsxnumtestcases query
%And a database of numpixelsxnumcases examples
%Sort the distances based on embedded distance
%Returns a sorted index (nearest neighbours first)
%Each row matches a row of features
%dg is a flag : 1 if we don't count the "diagonal" as a neighbour
% this is handy if we're passing in the training set as the query
% by default, dg=0
% requires GPU

batchsize=128; %batch size in which to do up-pass

if nargin<7
  dg=0;
end

[numpixels,numcases] = size(databasepixels);
nr=sqrt(numpixels);
assert(nr==floor(sqrt(numpixels))) %assert square
nc=nr;
[numpixels1,numtestcases] = size(querypixels);

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


%databasepixels_d = GPUsingle(databasepixels); %to GPU
%The up-pass is GPU memory intensive
%So we batch this up
z = zeros(numhid,numcases,GPUsingle); %to hold codes
numbatches=div((numcases+batchsize-1),batchsize); %round up
for bb=1:numbatches
  batchidx=(bb-1)*batchsize+1:min(bb*batchsize,numcases);
  data_d = GPUsingle(databasepixels(:,batchidx)); %avoids putting all data into GPU mem
  [y2,~,~,~,~,~] = convnet_forward2_gpu(data_d,filters1_d,b1_d, ...
    convcoeff1_d,downsample1,filters2_d, ...
    b2_d,convcoeff2_d,downsample2,connections,useabsrec);
  z(:,batchidx)=A_d*y2;
  %yy(:,batchidx)=y2;
end
clear data_d y2



%querypixels_d = GPUsingle(querypixels); %to GPU
zv = zeros(numhid,numtestcases,GPUsingle); %to hold codes
numbatches=div((numtestcases+batchsize-1),batchsize); %round up
for bb=1:numbatches
  batchidx=(bb-1)*batchsize+1:min(bb*batchsize,numtestcases);
  data_d = GPUsingle(querypixels(:,batchidx)); %avoids putting all data into GPU mem
  [y2,~,~,~,~,~] = convnet_forward2_gpu(data_d,filters1_d,b1_d, ...
    convcoeff1_d,downsample1,filters2_d, ...
    b2_d,convcoeff2_d,downsample2,connections,useabsrec);
  zv(:,batchidx)=A_d*y2;
  %yy(:,batchidx)=y2;
end
clear data_d y2


if numcases>numtestcases
  D=zeros(numtestcases,numcases,GPUsingle);

  cuDist(z,zv,D); %each thread computes a column (i.e. distance to one of
                 %the descriptors in the database
  
  D=transpose(D);
else
  D=zeros(numcases,numtestcases,GPUsingle);

  cuDist(zv,z,D); %each thread computes a column (i.e. distance to one of
                 %the descriptors in the database
end

%each column of D represents the distance of that validation case to each
%of the training points

Dh=single(D);

% if dg; Dh(1:numcases+1:end)=NaN; end  %So we don't match query image to itself
% [nv,nn] = min(Dh,[],1); %Nearest neighbour
% nv_d = GPUsingle(nv);

% GPUminus(D,repmat(nv_d,[numcases 1]),D); %subtract min element from
%                                              %every row
% GPUtimes(D,-1,D); %negative
% GPUexp(D,D); %exponent
    
% if dg; D(1:size(D,1)+1:end)=0; end %0 prob on diagonal

% denom = sum(D,1);
% GPUrdivide(D,repmat(denom,[numcases 1]),D); %this is p (probability
%                                             %of being neighbours)
% Dh=single(D); %back to GPU
% %We want the K nearest neighbours
% [nn_v,nn_idx] = sort(-Dh); %sort sorts each column

if dg
  if dg; Dh(1:numcases+1:end)=NaN; end  %So we don't match query image to itself
end
[nn_v,nn_idx] = sort(Dh); %sort sorts each column

nn_idx = nn_idx'; %since features are in columns
nn_v = nn_v'; %consistency
%[nv,nn] = min(Dh,[],1); %nn gives indices to the nearest neighbour in the training set
                        %idx = sub2ind([numcases numcasesvalid],nn,1:numcasesvalid); %linear indices
