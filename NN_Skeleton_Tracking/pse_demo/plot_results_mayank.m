% Plot nearest neighbour matches for some of the validation examples
% This works for either convolutional model, regardless of objective used
% (i.e. works for conv_ncareg or conv_drlim)
% Plots the best results first (this can be adjusted below)
cam = 'cam2';
% loadfile = ['../data/testdata/testdata_' cam '.mat'];
% load (loadfile);
 
%Load the original data (needed for visualization)
if ~exist('traindata','var') || ~exist('validdata','var')
  fprintf('Loading original (unnormalized) data.\n');
  %load('data/synthdata.mat','traindata','validdata');
  load(['data/init_data_' cam '.mat']);
end

%show some matches 
qc=size(validndata,3);%1024; %how many cases to go through
K=2; %# of nearest neighbours to show
usetrain=0; %use train instead of valid
bestfirst=1; %show best examples first (-1 means worst first)

[nr,nc,numcases]=size(trainndata); %use LCN data
numvalidcases=size(validndata,3);  %use LCN data

db = reshape(trainndata,nr*nc,numcases); %flatten to 2D

%Note that while the normalized (LCN) data is passed through the network
%to determine the embedding and NN
%We visualize using the original data (and therefore the labels
%correspond to the original data too)
if usetrain
  dg=1; %don't consider points on "diagonal"
  query = reshape(trainndata(:,:,1:qc),nr*nc,qc);
  queryimages = traindata(:,:,1:qc); %just for plotting
  querylabels = trainlabels(:,1:qc); %just for plotting

else
  dg=0; %consider all points when matching
  query = reshape(validndata(:,:,1:qc),nr*nc,qc);
  queryimages = validdata(:,:,1:qc); %just for plotting
  querylabels = validlabels(:,1:qc); %just for plotting
end

[nn_idx,nn_v] = get_neighbours(X,Dim,query,db,connections,useabsrec,dg);
 

%distance between nearest neighbour and each query image (in pose space)
%use image space rather than normalized space for matches and plots, etc.
bestd=sqrt(sum((querylabels-trainlabels(:,nn_idx(:,1))).^2,1));
%sort based on this distance
[bd,ix]=sort(bestd,2);

%Determine display order
if bestfirst==1
  ordr=ix;
elseif bestfirst==-1
  ordr=ix(end:-1:1);
else
  ordr=1:size(query,1);
end

fprintf('Top row: Pixel error for each neighbour\n');
fprintf('Bottom row: Distance in embedded space for each neighbour\n'); 

count = 1;
sum_err = 0;
for ii=ordr%1:size(query,1)
  fprintf('Query image: %d\n ',ii);
  disp([sqrt(sum(bsxfun(@minus,querylabels(:,ii),trainlabels(:,nn_idx(ii,1: ...
                                                    K))).^2));
  nn_v(ii,1:K)])
sum_err = sum_err + [sqrt(sum(bsxfun(@minus,querylabels(:,ii),trainlabels(:,nn_idx(ii,1: ...
                                                   K))).^2));
 nn_v(ii,1:K)];
count = count+1;
%   ultimateSubplot(2,1,1,[],0.05)
%   imshow(queryimages(:,:,ii));
%   %visualizePrediction(queryimages(:,:,ii),querylabels(:,ii),[])
%   title('query image')
%   
%   ultimateSubplot(2,1,2,[],0.05)
%   visualizePrediction(traindata(:,:,nn_idx(ii,1:K)),trainlabels(:,nn_idx(ii,1: ...
%                                                     K)),[])                                      
                                                
  title('nearest neighbours')
  fprintf('Press a key to see the next example\n');
%   pause;

end
count
sum_err/count
