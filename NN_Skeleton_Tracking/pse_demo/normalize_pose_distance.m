function p = normalize_pose_distance(y,T)

%given pose data, produce a normalized neighbour matrix
%for each column (i)
%pij is the probability that j is point i's neighbour

if nargin<2
  T=1;
end
numcases=size(y,2);
%Precompute normalized distance matrix in label space
%distance matrix in y (label) space
d_y = sdistmatbsxfun_t(y); %takes data as cols returns squared dist mat

%now we want to normalize d_y
d1 = d_y+diag(nan(1,numcases)); %NaN on diagonals
[nv,nn] = min(d1,[],1); %find nearest neighbour
d_y = bsxfun(@minus,d_y,nv); %make sure the thing we subtract doesn't get squared

negexpds = exp(-(d_y)/T); %note that d has already been squared
negexpds(1:size(d_y,1)+1:end)=0; %0 prob on diagonal

denom = sum(negexpds,1);

p = bsxfun(@rdivide,negexpds,denom); %note that p is based on the pose labels
