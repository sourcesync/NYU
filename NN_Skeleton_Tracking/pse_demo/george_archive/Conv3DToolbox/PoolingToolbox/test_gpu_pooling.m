%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% A test script for variosu gpu pooling and unpooling routines
% which compares speed for different types of pooling, different sizes of 
% images, and different numbers of images.
%
% @file
% @author Matthew Zeiler
% @date Apr 11, 2011
%
% @test @copybrief test_gpu_pooling.m
% @pooling_file @copybrief test_gpu_pooling.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

fprintf('XXXXXXXXXXXXXXXXXXXXXX\nLayer 1 Parameters 32 Images\nXXXXXXXXXXXXXXXXXXXXXX\n')
maps = randn(156,156,16,64,'single');
unpooled_size = size(maps);
% unpooled_size = unpooled_size(1:3); % Just like the real code.
gmaps = GPUsingle(maps);
pool_type = 'Avg3';
pool_size = [3 3 2];
fprintf('Computing on CPU\n');
GPUsync; tic; [pooled_maps,pooled_indices] = pool_wrapper(pool_type,maps,pool_size,[],4); tC=toc;
fprintf('CPU time for %s (%d x %d x %d): %f\n',pool_type,pool_size(1), pool_size(2),pool_size(3),tC);
fprintf('Computing on GPU\n');
GPUsync; tic; [gpooled_maps,gpooled_indices] = pool_wrapper(pool_type,gmaps,pool_size,[],4);GPUsync; t=toc;
fprintf('GPU time for %s (%d x %d x %d): %f, Speedup: %f\n',pool_type,pool_size(1), pool_size(2),pool_size(3),t,tC/t);
fprintf('Maximum Error in maps: %f\n\n',max(abs(pooled_maps(:)-single(gpooled_maps(:)))));
fprintf('Maximum Error in indices: %f\n\n',max(va(single(pooled_indices)-single(gpooled_indices))));
% % fprintf('Memory Free: %f\n',GPUmem)
% figure(1), sdf(pooled_maps)
% figure(2), sdf(gpooled_maps)
fprintf('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXx\n\n')
% pooled_maps
% gpooled_maps
% keyboard

fprintf('XXXXXXXXXXXXXXXXXXXXXX\nLayer 1 Parameters 32 Images Input Indices Pooling\nXXXXXXXXXXXXXXXXXXXXXX\n')
% maps = rand(156,156,15,32,'single');
% gmaps = GPUsingle(maps);
% pool_type = 'Max';
% pool_size = [3 3 2];
fprintf('Computing on CPU\n');
GPUsync; tic; [pooled_maps2,pooled_indices2] = pool_wrapper(pool_type,maps,pool_size,pooled_indices,4); tC=toc;
fprintf('CPU time for %s (%d x %d x %d): %f\n',pool_type,pool_size(1), pool_size(2),pool_size(3),tC);
fprintf('Computing on GPU\n');
GPUsync; tic; [gpooled_maps2,gpooled_indices2] = pool_wrapper(pool_type,gmaps,pool_size,gpooled_indices,4);GPUsync; t=toc;
fprintf('GPU time for %s (%d x %d x %d): %f, Speedup: %f\n',pool_type,pool_size(1), pool_size(2),pool_size(3),t,tC/t);
fprintf('Maximum Error in maps: %f\n\n',max(abs(pooled_maps2(:)-single(gpooled_maps2(:)))));
fprintf('Maximum Error in indices: %f\n\n',max(va(single(pooled_indices2)-single(gpooled_indices2))));
% fprintf('Memory Free: %f\n',GPUmem)
fprintf('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXx\n\n')

% pooled_indices2
% gpooled_indices2
% keyboard
fprintf('XXXXXXXXXXXXXXXXXXXXXX\nLayer 1 Parameters 32 Images UnPooling\nXXXXXXXXXXXXXXXXXXXXXX\n')
% maps = rand(156,156,15,32,'single');
% gmaps = GPUsingle(maps);
% pool_type = 'Max';
% pool_size = [3 3 2];
fprintf('Computing on CPU\n');
GPUsync; tic; [unpooled_maps] = unpool_wrapper(pool_type,pooled_maps,pooled_indices,pool_size,unpooled_size,4); tC=toc;
fprintf('CPU time for %s (%d x %d x %d): %f\n',pool_type,pool_size(1), pool_size(2),pool_size(3),tC);
fprintf('Computing on GPU\n');
GPUsync; tic; [gunpooled_maps] = unpool_wrapper(pool_type,gpooled_maps,gpooled_indices,pool_size,unpooled_size,4);GPUsync; t=toc;
fprintf('GPU time for %s (%d x %d x %d): %f, Speedup: %f\n',pool_type,pool_size(1), pool_size(2),pool_size(3),t,tC/t);
size(unpooled_maps)
size(gunpooled_maps)
% keyboard
fprintf('Maximum Error in maps: %f\n\n',max(abs(unpooled_maps(:)-single(gunpooled_maps(:)))));
fprintf('Maximum Error in indices: %f\n\n',max(abs(single(pooled_indices2(:))-single(gpooled_indices2(:)))));
fprintf('Maximum Error in sizes: %f\n\n',max(va(size(unpooled_maps)-size(gunpooled_maps))));
% fprintf('Memory Free: %f\n',GPUmem)
fprintf('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXx\n\n')


clear all




fprintf('XXXXXXXXXXXXXXXXXXXXXX\nLayer 4 Parameters 32 Images\nXXXXXXXXXXXXXXXXXXXXXX\n')
maps = randn(15,15,150,64,'single');
unpooled_size = size(maps);
unpooled_size = unpooled_size(1:3);
gmaps = GPUsingle(maps);
pool_type = 'Avg3';
pool_size = [3 3 2];
fprintf('Computing on CPU\n');
GPUsync; tic; [pooled_maps,pooled_indices] = pool_wrapper(pool_type,maps,pool_size,[],4); tC=toc;
fprintf('CPU time for %s (%d x %d x %d): %f\n',pool_type,pool_size(1), pool_size(2),pool_size(3),tC);
fprintf('Computing on GPU\n');
GPUsync; tic; [gpooled_maps,gpooled_indices] = pool_wrapper(pool_type,gmaps,pool_size,[],4);GPUsync; t=toc;
fprintf('GPU time for %s (%d x %d x %d): %f, Speedup: %f\n',pool_type,pool_size(1), pool_size(2),pool_size(3),t,tC/t);
fprintf('Maximum Error in maps: %f\n\n',max(abs(pooled_maps(:)-single(gpooled_maps(:)))));
fprintf('Maximum Error in indices: %f\n\n',max(va(single(pooled_indices)-single(gpooled_indices))));
% fprintf('Memory Free: %f\n',GPUmem)
fprintf('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXx\n\n')

fprintf('XXXXXXXXXXXXXXXXXXXXXX\nLayer 4 Parameters 32 Images Input Indices Pooling\nXXXXXXXXXXXXXXXXXXXXXX\n')
% maps = rand(156,156,15,32,'single');
% gmaps = GPUsingle(maps);
% pool_type = 'Max';
% pool_size = [3 3 2];
fprintf('Computing on CPU\n');
GPUsync; tic; [pooled_maps2,pooled_indices2] = pool_wrapper(pool_type,maps,pool_size,pooled_indices,4); tC=toc;
fprintf('CPU time for %s (%d x %d x %d): %f\n',pool_type,pool_size(1), pool_size(2),pool_size(3),tC);
fprintf('Computing on GPU\n');
GPUsync; tic; [gpooled_maps2,gpooled_indices2] = pool_wrapper(pool_type,gmaps,pool_size,gpooled_indices,4);GPUsync; t=toc;
fprintf('GPU time for %s (%d x %d x %d): %f, Speedup: %f\n',pool_type,pool_size(1), pool_size(2),pool_size(3),t,tC/t);
fprintf('Maximum Error in maps: %f\n\n',max(abs(pooled_maps2(:)-single(gpooled_maps2(:)))));
fprintf('Maximum Error in indices: %f\n\n',max(va(single(pooled_indices2)-single(gpooled_indices2))));
% fprintf('Memory Free: %f\n',GPUmem)
fprintf('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXx\n\n')



fprintf('XXXXXXXXXXXXXXXXXXXXXX\nLayer 4 Parameters 32 Images UnPooling\nXXXXXXXXXXXXXXXXXXXXXX\n')
% maps = rand(156,156,15,32,'single');
% gmaps = GPUsingle(maps);
% pool_type = 'Max';
% pool_size = [3 3 2];
fprintf('Computing on CPU\n');
GPUsync; tic; [unpooled_maps] = unpool_wrapper(pool_type,pooled_maps,pooled_indices,pool_size,unpooled_size,4); tC=toc;
fprintf('CPU time for %s (%d x %d x %d): %f\n',pool_type,pool_size(1), pool_size(2),pool_size(3),tC);
fprintf('Computing on GPU\n');
GPUsync; tic; [gunpooled_maps] = unpool_wrapper(pool_type,gpooled_maps,gpooled_indices,pool_size,unpooled_size,4);GPUsync; t=toc;
fprintf('GPU time for %s (%d x %d x %d): %f, Speedup: %f\n',pool_type,pool_size(1), pool_size(2),pool_size(3),t,tC/t);
fprintf('Maximum Error in maps: %f\n\n',max(abs(unpooled_maps(:)-single(gunpooled_maps(:)))));
fprintf('Maximum Error in sizes: %f\n\n',max(va(size(unpooled_maps)-size(gunpooled_maps))));
% fprintf('Memory Free: %f\n',GPUmem)
fprintf('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXx\n\n')




