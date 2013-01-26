function E = compute_pixel_error(nn,trainlabels,validlabels,skiphead)

%Compute error in pixel space
%nn is an index to nearest neighbours in the training set
%(same length as numcasesvalid) i.e. it indexes trainlabels
%trainlabels is the database labels
%validlabels is the query labels
%if skiphead is set, ignore the first two dimensions
if nargin<4
  skiphead=0; 
end

[numlabels,numcasesvalid] = size(validlabels);
assert(numcasesvalid==length(nn))

if skiphead %skip first two labels (i.e. if we only care about hands)
  labelidx = 3:numlabels;
else
  labelidx = 1:numlabels;
end

%Error in pixels
E = (trainlabels(labelidx,nn)-validlabels(labelidx,:)).^2;
E = reshape(E,[2 length(labelidx)/2 numcasesvalid]); %(x,y) first
E = squeeze(sum(E,1)); %sum over first dim (x,y)
E = sqrt(E); %sqrt gives Euclidean distance separately for each marker
E = mean(E(:)); %mean over markers and cases
