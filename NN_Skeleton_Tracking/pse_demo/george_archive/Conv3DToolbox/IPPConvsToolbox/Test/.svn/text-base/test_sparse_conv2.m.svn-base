%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% A test script for sparse_conv2.m which compares speed for different types of 
% convolutions, different sizes of images and filters, and different numbers
% of images and filters.
%
% @file
% @author Matthew Zeiler
% @date Apr 11, 2011
%
% @test @copybrief test_sparse_conv2.m
% @ipp_file @copybrief test_sparse_conv2.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% 
% 
% % z{1} = unpz{2};
% % z{1} = unpool_wrapper(model.norm_types{2},z{1},ones(size(pooled_indices{2}),'uint16'),model.norm_sizes{2},model.unpooled_sizes{2},4);
% % pz = pooled_wrapper(model.norm_types{2},z{1},
% 
% % keyboard
% for i=1:10
% %     [A] = CreateImages('/misc/FergusGroup/zeiler/Datasets/Images/city_fruit/','local_cn',0,'gray',1);
% % A = squeeze(A);
% A = reshape(z{1},[size(z{1},1) size(z{1},2) size(z{1},3)*size(z{1},4)]);
% 
% % A = randn(156,156,20,'single');
% % A(2:3:end,2:3:end,:) = 0;
% % A(3:3:end,3:3:end,:) = 0;
% % inds = randperm(numel(A));
% % inds = inds(1:floor(length(inds)/2));
% % A(inds) = randn(1,length(inds),'single');
% 
% % A(1,1) = 1;
% % A(1,2) = -1;
% F = randn(7,7,'single');
% 
% tic
% R1 = ipp_conv2(A,F,'valid');
% t1(i)=toc
% tic
% R2 = sparse_conv2(A,F,'valid');
% t2(i)=toc
% % R1
% % R2
% 
% norm(R2(:)-R1(:))
% max(abs(R2(:)-R1(:)))
% fprintf('ipp_conv2 time: %f, My time: %f, Speedup %f x\n',t1(i),t2(i),t1(i)/t2(i));
% 
% 
% end
% fprintf('Mean timing for ipp_conv2 vs sparse_conv2: IPP time: %f, My time: %f, Speedup %f x\n',mean(t1),mean(t2),mean(t1)/mean(t2));
% 






% keyboard
for i=1:10
%     [A] = CreateImages('/misc/FergusGroup/zeiler/Datasets/Images/city_fruit/','local_cn',0,'gray',1);
% A = squeeze(A);
% A = reshape(z{1},[size(z{1},1) size(z{1},2) size(z{1},3)*size(z{1},4)]);

A = zeros(156,156,4,10,'single');
% A(2:3:end,2:3:end,:) = 0;
% A(3:3:end,3:3:end,:) = 0;
inds = randperm(numel(A));
inds = inds(1:floor(length(inds)/18));
A(inds) = randn(1,length(inds),'single');

F = randn(7,7,3,4,10,'single');
C = ones(3,4);

tic
% Return the summed result.
R1 = valid_eachK_loopJ(A,F,C,4);
t1(i)=toc;
tic
R2 = sum(sparse_valid_eachK_loopJ(A,F,C,4),4);
t2(i)=toc;
% R1
% R2

norm(R2(:)-R1(:))
max(abs(R2(:)-R1(:)))
fprintf('valid_eachK_loopJ time: %f, My time: %f, Speedup %f x\n',t1(i),t2(i),t1(i)/t2(i));


end
fprintf('Mean timing for valid_eachK_loopJ vs sparse_valid_eachK_loopJ: IPP time: %f, My time: %f, Speedup %f x\n',mean(t1),mean(t2),mean(t1)/mean(t2));





