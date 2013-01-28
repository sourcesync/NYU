%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% A test script for variosu gpu cuda pooling and unpooling routines
% which compares speed for different types of pooling, different sizes of 
% images, and different numbers of images.
%
% @file
% @author Matthew Zeiler
% @date Apr 29, 2011
%
% @test @copybrief testcuMaxPool.m
% @pooling_file @copybrief testcuMaxPool.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
A =randn(52,52,50,64,'single');
psize = [3 3];

gA = GPUsingle(A);

GPUsync;
tic
[gC,gI] = cuMaxPool(gA,psize);
GPUsync;
tg = toc

tic
[C,I] = max_pool(A,psize,[],4);
tc = toc

fprintf('Speedup when No indices are passed in: %f\n',tc/tg);
fprintf('Max Abs Diff: %f\n',max(abs(single(gC(:))-C(:))));
fprintf('Max Abs Diff Ind: %f\n',max(abs(single(gI(:))-single(I(:)))));


GPUsync;
tic
[gC,gI] = cuMaxPool(gA,psize,gI);
GPUsync;
tg = toc

tic
[C,I] = max_pool(A,psize,I,4);
tc = toc

fprintf('Speedup when No indices are passed in: %f\n',tc/tg);
fprintf('Max Abs Diff: %f\n',max(abs(single(gC(:))-C(:))));
fprintf('Max Abs Diff Ind: %f\n',max(abs(single(gI(:))-single(I(:)))));




