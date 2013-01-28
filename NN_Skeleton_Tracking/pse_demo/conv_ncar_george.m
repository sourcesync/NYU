%george
addpath( genpath( './george_extras' ) );
GPUstart
%george


%Convolutional net settings (filter sizes, pooling factors, etc.)
filtersize1=9;
nummaps1=16;
downsample1=5;

filtersize2=9;
nummaps2=32;
downsample2=4;

useabsrec=1; %use abs rectification?

numhid = 32; %ultimately project down to this dimension

%connectivity map needed only for the 2nd layer (i.e. first layer is
%connected only to a single map - the input) each feature map in layer 1 is
%connected to the input
num_connect = 4; 

weightdecay=0.0002;
maxlinesearches=3; %do this many CG line searches for each mini-batch
batchsize=192; %size of mini-batches (multiple of 16 for GPU)
numbatches=2000; %
                 
%numruns=100;
skiphead=0; % compute error on head as well as hands

snapshot_path = '/tmp/conv_ncar'; %change this to where weights
                                        %should be written
snapshotevery = 100; %how often (number of mini-batches) to do snapshots

if ~exist(snapshot_path,'dir')
  if isunix
    %create directory
    fprintf('Creating %s\n',snapshot_path);
    system(['mkdir -p ' snapshot_path]);
  else
    error('Create snapshot path');
  end
end

%now copy this file to the path, so we have a record of the experiment
if isunix
  copystring = sprintf('cp ./conv_ncar_george.m %s',snapshot_path);
  assert(system(copystring)==0);
end

%addpath('/home/gwtaylor/matlab/bregler/greendot/hashing_tinypeople')
fprintf('Loading data.\n');
load('data/synthdata_lcn.mat','trainndata','trainlabels_r','trainlabels','validndata','validlabels_r','validlabels');

[nr,nc,numcases]=size(trainndata); %use LCN data
data=reshape(trainndata,nr*nc,numcases); %flatten to 2D

numvalidcases=size(validndata,3);
vd=reshape(validndata,nr*nc,numvalidcases); %flatten to 2D

assert(nr==nc) %square data assumption (GPU convolution routines)

numlabels = size(trainlabels_r,1); %use relative labels when training
                                   %(i.e. hands are relative to head)

%dimensions of convolutional layer 1
outr1=nr-filtersize1+1;
outc1=nc-filtersize1+1;
%dimensions of subsampling layer 1
nr1 = outr1/downsample1; nc1 = outc1/downsample1; %downsampled dimensions

%dimensions of convolutional layer 2
outr2=nr1-filtersize2+1;
outc2=nc1-filtersize2+1;

%dimensions of subsampling layer 2
nr2 = outr2/downsample2; nc2 = outc2/downsample2;

outsize=nr2*nc2*nummaps2; %classification weights are connected to all
                          %maps in the second subsampling layer

%fid=fopen(sprintf('%s/log.txt',snapshot_path),'a');

%for ii=1:numruns

%parameter initialization
%filters1 = 0.01*randn(filtersize1*filtersize1,nummaps1);
filters1 = -0.11+0.22*rand(filtersize1*filtersize1,nummaps1);
b1 = 0.01*randn(1,nummaps1); %trainable bias, inside tanh

convcoeff1 = ones(nummaps1,1) + 0.01*randn(nummaps1,1); %trainable scalar coeff 1 per map
                                                        
%filters2 = 0.01*randn(filtersize2*filtersize2,num_connect*nummaps2);
filters2 = -0.11+0.22*rand(filtersize2*filtersize2,num_connect*nummaps2);
b2 = 0.01*randn(1,nummaps2); %trainable bias, inside tanh
  
convcoeff2 = ones(nummaps2,1)+ 0.01*randn(nummaps2,1); %trainable scalar coeff 1 per map
  
A = -0.11+0.22*rand(numhid,outsize); %no bias to code layer

% CONNECTIVITY
% Each map of second convolutional layer is randomly connected to 4 maps
% from the first subsampling output map
% each row in connections corresponds to a feature map in the second
% convolution layer
connections = zeros(nummaps2,num_connect);
%rand('state',0);
for jj=1:nummaps2
  %slightly ugly because it uses the stats toolbox to sample without
  %replacement
  connections(jj,:) = randsample(nummaps1,num_connect); 
end

Dim(1)=filtersize1;
Dim(2)=nummaps1;
Dim(3)=downsample1;
Dim(4)=filtersize2;
Dim(5)=nummaps2;
Dim(6)=downsample2;
Dim(7)=outsize;
Dim(8)=numhid;
  
cc=0;

fprintf('Starting learning. %d mini-batches. Reporting error and saving weights every %d mini-batches.\n',numbatches,snapshotevery);
fprintf('This will now output:\n');
fprintf(['Mini-batch number; Training cost (NCAR objective); Validation ' ...
         'cost (NCAR objective); Validation cost (Pixel error)\n']);
for bb=1:numbatches %process this many random mini-batches
                    
  %read in a batch
  batchrndidx = randSample(numcases,batchsize); %without replacement
  
  %parameters are flattened into a vector
  vv = [filters1(:);b1(:);convcoeff1(:);filters2(:);b2(:);convcoeff2(:);A(:)];

  %Do line searches for this batch
  %Note this uses Carl Rasmussen's minimize to do CG optimization
  %See his documentation & examples for more info
  [X,fX,i]=minimize(vv,'convnetncareg',maxlinesearches,Dim, ...
      data(:,batchrndidx),trainlabels_r(:,batchrndidx),connections, ...
      weightdecay,useabsrec);

  filters1 = reshape(X(1:filtersize1*filtersize1*nummaps1),[filtersize1*filtersize1 nummaps1]);
  xxx = filtersize1*filtersize1*nummaps1;
  b1 = reshape(X(xxx+1:xxx+nummaps1),1,nummaps1);
  xxx = xxx+nummaps1;
  convcoeff1 = reshape(X(xxx+1:xxx+nummaps1),nummaps1,1);
  xxx = xxx+nummaps1;
  filters2 = reshape(X(xxx+1:xxx+filtersize2*filtersize2*(num_connect*nummaps2)),[filtersize2*filtersize2 num_connect*nummaps2]);
  xxx = xxx+filtersize2*filtersize2*(num_connect*nummaps2);
  b2 = reshape(X(xxx+1:xxx+nummaps2),1,nummaps2);
  xxx=xxx+nummaps2;
  convcoeff2 = reshape(X(xxx+1:xxx+nummaps2),nummaps2,1);
  xxx = xxx+nummaps2;
  A = reshape(X(xxx+1:xxx+numhid*outsize),numhid,outsize);
    
  if mod(bb,snapshotevery)==0
      cc=cc+1;
      %report error
      %[err_train,err_valid,err_validmax] = convnetncaregeval_gpu(X,Dim, ...
      %                                                  data,trainlabels_r,vd,validlabels_r,connections,useabsrec);
      %Use the true labels (not relative labels) to measure pixel error
      [err_train,err_valid,err_validmax,err_pix] = convnetncaregeval_gpu(X,Dim,data,trainlabels,vd,validlabels,connections,useabsrec,skiphead);
      e1(cc)=err_train; e2(cc)=err_valid; e3(cc)=err_pix;
      xx(cc)=bb;
      fprintf(1,'%d %6.4f %6.4f %6.4f\n',bb,e1(cc),e2(cc),e3(cc));
      %fprintf(fid,'%d %d %6.4f %6.4f %6.4f\n',ii,bb,e1(cc),e2(cc),e3(cc));  
      %save snapshot
      snapshot_file=sprintf('%s/conv_ncar_snapshot_batch%d.mat',snapshot_path,bb);
      save(snapshot_file,'X','Dim','connections','useabsrec','e1','e2','e3','xx','cc');      
  end
end

%fclose(fid);

