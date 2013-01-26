function visualizePrediction(X,ytrue,y,h,w)

%takes a stack of images (or a 3D array), X
%takes the true labels, ytrue
%takes the predicted labels, y (this can also be empty)
%visualizes the predictions
%if we have a prediction, but no true labels, pass in [] for ytrue
if nargin<3
  y=[]; %no prediction
end

if ndims(X)==3
  %make a stack
  [h,w,numcases]=size(X);
  X = reshape(X,h*w,numcases);
elseif ndims(X)==2
  if nargin<5
    %fprintf('Assuming singleton image (for stack must pass in h,w)\n');
    [h,w] = size(X);
    X = reshape(X,h*w,1);
  end
else
  error('X must be 2 or 3 dims');
end

[numpixels,numcases] = size(X);

%make sure image is double
X = double(X);

imdisp = dispimslab(X,ytrue,y,h,w,0,2);
%imdisp = dispimslab(X,ytrue,y,h,w,0,2,1); %single row plot

function [imdisp] = dispimslab(imstack,ytrue,y,drows,dcols,flip,border,n2,fud)
% [imdisp] = dispims(imstack,drows,dcols,flip,border,frame_rows)
%
% display a stack of images

[pp,N] = size(imstack);
if(nargin<9) fud=0; end
if(nargin<8) n2=ceil(sqrt(N)); end

if(nargin<5) dcols=drows; end
if(nargin<6) flip=0; end
if(nargin<7) border=2; end

drb=drows+border;
dcb=dcols+border;

imdisp=min(imstack(:))+zeros(n2*drb,ceil(N/n2)*dcb);

for nn=1:N

  ii=rem(nn,n2); if(ii==0) ii=n2; end
  jj=ceil(nn/n2);

  if(flip)
    daimg = reshape(imstack(:,nn),dcols,drows)';
  else
    daimg = reshape(imstack(:,nn),drows,dcols);
  end

%  daimg=daimg/max(daimg(:)); %% RESCALING IN DISPIMSC
  
  imdisp(((ii-1)*drb+1):(ii*drb-border),((jj-1)*dcb+1):(jj*dcb-border))=daimg;

end

if(fud)
imdisp=flipud(imdisp);
end

imagesc(imdisp); colormap gray; axis equal; axis off;
drawnow; hold on;

%now plot labels
for nn=1:N
    ii=rem(nn,n2); if(ii==0) ii=n2; end
    jj=ceil(nn/n2);
    
    %note that labels are x first (i.e. cols)
    %so we need to switch the order of the adjustment
    if(ytrue);plot((jj-1)*dcb++ytrue(1,nn),(ii-1)*drb+ytrue(2,nn),'mo');end
    if(y);plot((jj-1)*dcb++y(1,nn),(ii-1)*drb+y(2,nn),'m+');end
    
    if(ytrue);plot((jj-1)*dcb++ytrue(3,nn),(ii-1)*drb+ytrue(4,nn),'co');end
    if(y);plot((jj-1)*dcb++y(3,nn),(ii-1)*drb+y(4,nn),'c+');end
    
    if(ytrue);plot((jj-1)*dcb++ytrue(5,nn),(ii-1)*drb+ytrue(6,nn),'go');end
    if(y);plot((jj-1)*dcb++y(5,nn),(ii-1)*drb+y(6,nn),'g+');end
    
end

  

