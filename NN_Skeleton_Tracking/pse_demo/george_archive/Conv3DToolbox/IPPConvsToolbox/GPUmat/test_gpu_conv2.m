%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% A test script for gpu_conv2.m, valid_each3_sum4_gpu.m, valid_each3_each4_gpu.m
% and full_each4_sum3_gpu.m which compares speed for different types of 
% convolutions, different sizes of images and filters, and different numbers
% of images and filters.
%
% @file
% @author Matthew Zeiler
% @date Apr 11, 2011
%
% @test @copybrief test_gpu_conv2.m
% @gpu_file @copybrief test_gpu_conv2.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% clear all
% 
% fprintf('XXXXXXXXXXXXXXXXXXXXXX\nLayer 1 Parameters\nXXXXXXXXXXXXXXXXXXXXXX\n')
% maps = randn(156,156,15,'single');
% gmaps = GPUsingle(maps);
% F = randn(7,7,3,15,'single');
% % F(:,:,2,:) = 2;
% gF  =GPUsingle(F);
% C = ones(3,15,'single');
% gC = GPUsingle(C);
% fprintf('Computing on CPU\n');
% tCPU =tic;
% R = valid_each3_sum4(maps,F,C,4);
% tC=toc(tCPU);
% fprintf('CPU time for valid_each3_sum4: %f\n',tC);
% fprintf('Computing on GPU\n');
% tGPU=tic; gR = valid_each3_sum4(gmaps,gF,gC,4); GPUsync; t=toc(tGPU);
% fprintf('GPU time for valid_each3_sum4: %f, Speedup: %f\n',t,tC/t);
% fprintf('Maximum Error: %f\n',max(abs(R(:)-single(gR(:)))));
% % fprintf('Memory Free: %f\n',GPUmem)
% fprintf('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXx\n\n')
% size(F)
% size(gF)
% size(maps)
% size(gmaps)
% size(R)
% size(gR)
% % figure(1), sdf(R);
% % figure(2), sdf(single(gR));
% % squeeze(R)
% % squeeze(single(gR))
% clear all
% 
% 
% 
% fprintf('XXXXXXXXXXXXXXXXXXXXXX\nLayer 4 Parameters\nXXXXXXXXXXXXXXXXXXXXXX\n')
% maps = randn(15,15,150,'single');
% gmaps = GPUsingle(maps);
% F = randn(7,7,100,150,'single');
% gF  =GPUsingle(F);
% C = ones(100,150,'single');
% gC = GPUsingle(C);
% fprintf('Computing on CPU\n');
% tCPU=tic;
% R = valid_each3_sum4(maps,F,C,4);
% tC=toc(tCPU);
% fprintf('CPU time for valid_each3_sum4: %f\n',tC);
% fprintf('Computing on GPU\n');
% % profile on
% tGPU = tic; 
% gR = valid_each3_sum4(gmaps,gF,gC,4);
%  GPUsync; t=toc(tGPU);
% % profile viewer
% fprintf('GPU time for valid_each3_sum4: %f, Speedup: %f\n',t,tC/t);
% fprintf('Maximum Error: %f\n',max(abs(R(:)-single(gR(:)))));
% % fprintf('Memory Free: %f\n',GPUmem)
% fprintf('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXx\n\n')
% size(F)
% size(gF)
% size(maps)
% size(gmaps)
% clear all
% 
% 
% 
% fprintf('XXXXXXXXXXXXXXXXXXXXXX\nLayer 1 Parameters 37 Images\nXXXXXXXXXXXXXXXXXXXXXX\n')
% maps = randn(156,156,16,37,'single');
% gmaps = GPUsingle(maps);
% F = randn(7,7,2,16,'single');
% gF  =GPUsingle(F);
% C = ones(2,16,'single');
% gC = GPUsingle(C);
% fprintf('Computing on CPU\n');
% tic;
% R = valid_each3_sum4(maps,F,C,4);
% tC=toc;
% fprintf('CPU time for valid_each3_sum4: %f\n',tC);
% fprintf('Computing on GPU\n');
% TGPU=tic; gR = valid_each3_sum4(gmaps,gF,gC,4); GPUsync; t=toc(TGPU);
% % figure(1),sdf(squeeze(R));
% % figure(2), sdf(squeeze(single(gR)));
% % profile viewer
% fprintf('GPU time for valid_each3_sum4: %f, Speedup: %f\n',t,tC/t);
% fprintf('Maximum Error: %f\n',max(abs(R(:)-single(gR(:)))));
% % fprintf('Memory Free: %f\n',GPUmem)
% fprintf('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXx\n\n')
% size(F)
% size(gF)
% size(maps)
% size(gmaps)
% clear all
% % % 
% 
% fprintf('XXXXXXXXXXXXXXXXXXXXXX\nLayer 4 Parameters 37 Images\nXXXXXXXXXXXXXXXXXXXXXX\n')
% maps = randn(15,15,150,37,'single');
% gmaps = GPUsingle(maps);
% F = randn(7,7,100,150,'single');
% gF  =GPUsingle(F);
% C = ones(100,150,'single');
% gC = GPUsingle(C);
% fprintf('Computing on CPU\n');
% tic;
% R = valid_each3_sum4(maps,F,C,4);
% tC=toc;
% fprintf('CPU time for valid_each3_sum4: %f\n',tC);
% fprintf('Computing on GPU\n');
% TGPU=tic; gR = valid_each3_sum4(gmaps,gF,gC,4); GPUsync; t=toc(TGPU);
% % figure(1),sdf(squeeze(R));
% % figure(2), sdf(squeeze(single(gR)));
% % profile viewer
% fprintf('GPU time for valid_each3_sum4: %f, Speedup: %f\n',t,tC/t);
% fprintf('Maximum Error: %f\n',max(abs(R(:)-single(gR(:)))));
% % fprintf('Memory Free: %f\n',GPUmem)
% fprintf('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXx\n\n')
% size(F)
% size(gF)
% size(maps)
% size(gmaps)
% clear all
% 
% fprintf('XXXXXXXXXXXXXXXXXXXXXX\nLayer 1 Parameters 37 Images, Different Filters\nXXXXXXXXXXXXXXXXXXXXXX\n')
% maps = randn(156,156,16,37,'single');
% gmaps = GPUsingle(maps);
% F = randn(7,7,3,16,37,'single');
% gF  =GPUsingle(F);
% C = ones(3,16,'single');
% gC = GPUsingle(C);
% fprintf('Computing on CPU\n');
% tic;
% R = valid_each3_sum4(maps,F,C,4);
% tC=toc;
% fprintf('CPU time for valid_each3_sum4: %f\n',tC);
% fprintf('Computing on GPU\n');
% TGPU=tic; gR = valid_each3_sum4(gmaps,gF,gC,4); GPUsync; t=toc(TGPU);
% % figure(1),sdf(squeeze(R));
% % figure(2), sdf(squeeze(single(gR)));
% % profile viewer
% fprintf('GPU time for valid_each3_sum4: %f, Speedup: %f\n',t,tC/t);
% fprintf('Maximum Error: %f\n',max(abs(R(:)-single(gR(:)))));
% % fprintf('Memory Free: %f\n',GPUmem)
% fprintf('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXx\n\n')
% % figure(1), sdf(R);
% % figure(2), sdf(single(gR));
% size(F)
% size(gF)
% size(maps)
% size(gmaps)
% 
% % figure(1), sdf(R-single(gR));
% % keyboard
% clear all
% 
% 
% 
% fprintf('XXXXXXXXXXXXXXXXXXXXXX\nLayer 4 Parameters 37 Images, Different Filters\nXXXXXXXXXXXXXXXXXXXXXX\n')
% maps = randn(15,15,150,37,'single');
% gmaps = GPUsingle(maps);
% F = randn(7,7,100,150,37,'single');
% gF  =GPUsingle(F);
% C = ones(100,150,'single');
% gC = GPUsingle(C);
% fprintf('Computing on CPU\n');
% tic;
% R = valid_each3_sum4(maps,F,C,4);
% tC=toc;
% fprintf('CPU time for valid_each3_sum4: %f\n',tC);
% fprintf('Computing on GPU\n');
% TGPU=tic; gR = valid_each3_sum4(gmaps,gF,gC,4); GPUsync; t=toc(TGPU);
% % figure(1),sdf(squeeze(R));
% % figure(2), sdf(squeeze(single(gR)));
% % profile viewer
% fprintf('GPU time for valid_each3_sum4: %f, Speedup: %f\n',t,tC/t);
% fprintf('Maximum Error: %f\n',max(abs(R(:)-single(gR(:)))));
% % fprintf('Memory Free: %f\n',GPUmem)
% fprintf('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXx\n\n')
% size(F)
% size(gF)
% size(maps)
% size(gmaps)
% % figure(1), sdf(R)
% % figure(2), sdf(single(gR))
% clear all
% 
% 
% 
% 
% 
% 
% 
% fprintf('XXXXXXXXXXXXXXXXXXXXXX\nLayer 1 Parameters 37 Images Sparse Connectivity\nXXXXXXXXXXXXXXXXXXXXXX\n')
% maps = randn(156,156,16,37,'single');
% gmaps = GPUsingle(maps);
% F = randn(7,7,3,16,'single');
% gF  =GPUsingle(F);
% % C = ones(2,16,'single');
% C = single(conmat_randdoub(3,16));
% gC = GPUsingle(C);
% fprintf('Computing on CPU\n');
% tic;
% R = valid_each3_sum4(maps,F,C,4);
% tC=toc;
% fprintf('CPU time for valid_each3_sum4: %f\n',tC);
% fprintf('Computing on GPU\n');
% TGPU=tic; gR = valid_each3_sum4(gmaps,gF,gC,4); GPUsync; t=toc(TGPU);
% % figure(1),sdf(squeeze(R));
% % figure(2), sdf(squeeze(single(gR)));
% % profile viewer
% fprintf('GPU time for valid_each3_sum4: %f, Speedup: %f\n',t,tC/t);
% fprintf('Maximum Error: %f\n',max(abs(R(:)-single(gR(:)))));
% % fprintf('Memory Free: %f\n',GPUmem)
% fprintf('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXx\n\n')
% size(F)
% size(gF)
% size(maps)
% size(gmaps)
% clear all
% 
% 
% fprintf('XXXXXXXXXXXXXXXXXXXXXX\nLayer 4 Parameters 37 Images Sparse Connectivity\nXXXXXXXXXXXXXXXXXXXXXX\n')
% maps = randn(15,15,150,37,'single');
% gmaps = GPUsingle(maps);
% F = randn(7,7,100,150,'single');
% gF  =GPUsingle(F);
% % C = ones(100,150,'single');
% C = single(conmat_randdoub(100,150));
% gC = GPUsingle(C);
% fprintf('Computing on CPU\n');
% tic;
% R = valid_each3_sum4(maps,F,C,4);
% tC=toc;
% fprintf('CPU time for valid_each3_sum4: %f\n',tC);
% fprintf('Computing on GPU\n');
% TGPU=tic; gR = valid_each3_sum4(gmaps,gF,gC,4); GPUsync; t=toc(TGPU);
% % figure(1),sdf(squeeze(R));
% % figure(2), sdf(squeeze(single(gR)));
% % profile viewer
% fprintf('GPU time for valid_each3_sum4: %f, Speedup: %f\n',t,tC/t);
% fprintf('Maximum Error: %f\n',max(abs(R(:)-single(gR(:)))));
% % fprintf('Memory Free: %f\n',GPUmem)
% fprintf('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXx\n\n')
% size(F)
% size(gF)
% size(maps)
% size(gmaps)
% clear all








% % % % % % % % 
% % % % % % % % % fprintf('XXXXXXXXXXXXXXXXXXXXXX\nPermuted Layer 1 Parameters 37 Images\nXXXXXXXXXXXXXXXXXXXXXX\n')
% % % % % % % % % maps = randn(156,156,15,37,'single');
% % % % % % % % % % temp = reshape(maps,[7 7 4 3]);
% % % % % % % % % % temp = permute(temp,[1 2 4 3]);
% % % % % % % % % % temp = reshape(temp,[7 7 4 3]);
% % % % % % % % % gmaps = GPUsingle(maps);
% % % % % % % % % F = zeros(7,7,3,15,'single');
% % % % % % % % % F(1,1) = 1;
% % % % % % % % % gF  =GPUsingle(F);
% % % % % % % % % C = ones(3,15,'single');
% % % % % % % % % gC = GPUsingle(C);
% % % % % % % % % fprintf('Computing on CPU\n');
% % % % % % % % % tic;
% % % % % % % % % R = valid_each3_sum4(maps,F,C,4);
% % % % % % % % % tC=toc;
% % % % % % % % % fprintf('CPU time for valid_each3_sum4: %f\n',tC);
% % % % % % % % % fprintf('Computing on GPU\n');
% % % % % % % % % TGPU=tic; gR = valid_each3_sum4(gmaps,gF,gC,4); GPUsync; t=toc(TGPU);
% % % % % % % % % % figure(1),sdf(squeeze(R));
% % % % % % % % % % figure(2), sdf(squeeze(single(gR)));
% % % % % % % % % % profile viewer
% % % % % % % % % fprintf('GPU time for valid_each3_sum4: %f, Speedup: %f\n',t,tC/t);
% % % % % % % % % fprintf('Maximum Error: %f\n',max(abs(va(R)-single(va(gR)))));
% % % % % % % % % fprintf('Memory Free: %f\n',GPUmem)
% % % % % % % % % fprintf('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXx\n\n')
% % % % % % % % % size(F)
% % % % % % % % % size(gF)
% % % % % % % % % size(maps)
% % % % % % % % % size(gmaps)
% % % % % % % % % figure(1), sdf(R);
% % % % % % % % % % figure(2), sdf(single(permute(gR,[1 2 4 3])));
% % % % % % % % % figure(3), sdf(single(gR));
% % % % % % % % % size(R)
% % % % % % % % % size(gR)
% % % % % % % % % R(1:10,1:10,1,1)
% % % % % % % % % gR(1:10,1:10,1,1)
% % % % % % % % % gR = permute(gR,[1 2 4 3]);
% % % % % % % % % gR(1:10,1:10,1,1)
% % % % % % % % % % clear all
% 
% 
% 
% 
% 
% 
% 




fprintf('XXXXXXXXXXXXXXXXXXXXXX\nLayer 1 Parameters Single Image\nXXXXXXXXXXXXXXXXXXXXXX\n')
maps = randn(156,156,15,'single');
gmaps = GPUsingle(maps);
F = randn(150,150,1,'single');
gF  =GPUsingle(F);
C = ones(1,15,'single');
gC = GPUsingle(C);
fprintf('Computing on CPU\n');
tic;R = valid_each3_each4(maps,F,C,4);tC=toc;
fprintf('CPU time for valid_each3_each4: %f\n',tC);
fprintf('Computing on GPU\n');
% gprofile on
tic; gR = valid_each3_each4(gmaps,gF,gC,4); GPUsync; t=toc;
% gprofile report
fprintf('GPU time for valid_each3_each4: %f, Speedup: %f\n',t,tC/t);
% figure(1), sdf(R);
% figure(2), sdf(single(gR));
fprintf('Maximum Error: %f\n\n',max(abs(R(:)-single(gR(:)))));
% fprintf('Memory Free: %f\n',GPUmem)
fprintf('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXx\n\n')
size(R)
size(gR)

% % keyboard/
% clear all
% 
% fprintf('XXXXXXXXXXXXXXXXXXXXXX\nLayer 4 Parameters Single Image\nXXXXXXXXXXXXXXXXXXXXXX\n')
% maps = randn(15,15,150,'single');
% gmaps = GPUsingle(maps);
% F = randn(9,9,100,'single');
% % F(1,1) = 1;
% % F(1,1,end,end) = 1;
% gF  =GPUsingle(F);
% C = ones(100,150,'single');
% gC = GPUsingle(C);
% fprintf('Computing on CPU\n');
% tic;R = valid_each3_each4(maps,F,C,4);tC=toc;
% fprintf('CPU time for valid_each3_each4: %f\n',tC);
% fprintf('Computing on GPU\n');
% tic; gR = valid_each3_each4(gmaps,gF,gC,4); GPUsync; t=toc;
% fprintf('GPU time for valid_each3_each4: %f, Speedup: %f\n',t,tC/t);
% sR = size(R);
% % figure(1), sdf(reshape(R,[sR(1:3) prod(sR(4:5))]));
% % figure(2), sdf(single(gR));
% fprintf('Maximum Error: %f\n\n',max(abs(R(:)-single(gR(:)))));
% % fprintf('Memory Free: %f\n',GPUmem)
% fprintf('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXx\n\n')
% size(R)
% size(gR)
% % figure(1),clf, sdf(R);
% % figure(2),clf, sdf(single(gR));
% clear all
% 
% 
% fprintf('XXXXXXXXXXXXXXXXXXXXXX\nLayer 1 Parameters 37 images\nXXXXXXXXXXXXXXXXXXXXXX\n')
% maps = ones(156,156,15,37,'single');
% gmaps = GPUsingle(maps);
% F = randn(150,150,3,37,'single');
% gF  =GPUsingle(F);
% C = ones(3,15,'single');
% gC = GPUsingle(C);
% fprintf('Computing on CPU\n');
% tic;R = valid_each3_each4(maps,F,C,4);tC=toc;
% fprintf('CPU time for valid_each3_each4: %f\n',tC);
% fprintf('Computing on GPU\n');
% tic; gR = valid_each3_each4(gmaps,gF,gC,4); GPUsync; t=toc;
% fprintf('GPU time for valid_each3_each4: %f, Speedup: %f\n',t,tC/t);
% % figure, sdf(R);
% % figure, sdf(single(gR));
% fprintf('Maximum Error: %f\n\n',max(abs(R(:)-single(gR(:)))));
% % fprintf('Memory Free: %f\n',GPUmem)
% fprintf('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXx\n\n')
% size(R)
% sR = size(R);
% size(gR)
% % R
% % fprintf('GPUreslt\n')
% % single(gR)
% % figure(1), sdf(reshape(R,[sR(1:3) prod(sR(4:5))]));
% % figure(2), sdf(reshape(single(R),[sR(1:3) prod(sR(4:5))]));
% % clear all
% 
% fprintf('XXXXXXXXXXXXXXXXXXXXXX\nLayer 4 Parameters 37 images\nXXXXXXXXXXXXXXXXXXXXXX\n')
% maps = randn(15,15,150,37,'single');
% gmaps = GPUsingle(maps);
% F = randn(9,9,100,37,'single');
% gF  =GPUsingle(F);
% C = ones(100,150,'single');
% gC = GPUsingle(C);
% fprintf('Computing on CPU\n');
% tic;R = valid_each3_each4(maps,F,C,4);tC=toc;
% fprintf('CPU time for valid_each3_each4: %f\n',tC);
% fprintf('Computing on GPU\n');
% tic; gR = valid_each3_each4(gmaps,gF,gC,4); GPUsync; t=toc;
% fprintf('GPU time for valid_each3_each4: %f, Speedup: %f\n',t,tC/t);
% % figure, sdf(R);
% % figure, sdf(single(gR));
% fprintf('Maximum Error: %f\n\n',max(abs(R(:)-single(gR(:)))));
% % fprintf('Memory Free: %f\n',GPUmem)
% fprintf('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXx\n\n')
% size(F)
% size(gF)
% size(maps)
% size(gmaps)
% clear all
% 
% 
% 
% 
% 
% 
% 
% fprintf('XXXXXXXXXXXXXXXXXXXXXX\nLayer 1 Parameters 37 images, Sparse Connectivity\nXXXXXXXXXXXXXXXXXXXXXX\n')
% maps = ones(156,156,15,5,'single');
% gmaps = GPUsingle(maps);
% F = randn(150,150,3,5,'single');
% gF  =GPUsingle(F);
% % C = ones(3,15,'single');
% C = single(conmat_randdoub(3,15));
% gC = GPUsingle(C);
% fprintf('Computing on CPU\n');
% tic;R = valid_each3_each4(maps,F,C,4);tC=toc;
% fprintf('CPU time for valid_each3_each4: %f\n',tC);
% fprintf('Computing on GPU\n');
% tic; gR = valid_each3_each4(gmaps,gF,gC,4); GPUsync; t=toc;
% fprintf('GPU time for valid_each3_each4: %f, Speedup: %f\n',t,tC/t);
% % figure, sdf(R);
% % figure, sdf(single(gR));
% fprintf('Maximum Error: %f\n\n',max(abs(R(:)-single(gR(:)))));
% % fprintf('Memory Free: %f\n',GPUmem)
% fprintf('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXx\n\n')
% size(R)
% sR = size(R);
% size(gR)
% % R
% % fprintf('GPUreslt\n')
% % figure(1), sdf(reshape(R,[sR(1:3) prod(sR(4:5))]));
% % figure(2), sdf(reshape(single(gR),[sR(1:3) prod(sR(4:5))]));
% 
% % 
% 
% % figure(1); clf; sdf(permute(reshape(R,[sR(1:3) prod(sR(4:5))]),[1 2 4 3]));
% % figure(2); clf; sdf(permute(reshape(single(gR),[sR(1:3) prod(sR(4:5))]),[1 2 4 3]));
% % % clear all
% % temp = single(gR);
% % temp = bsxfun(@times,reshape(C,[1 1 size(C,1) size(C,2) 1]),temp);
% % gR = temp;
% % fprintf('Maximum Error not considering C==0: %f\n\n',max(abs(R(:)-single(gR(:)))));
% % keyboard
% 
% 
% fprintf('XXXXXXXXXXXXXXXXXXXXXX\nLayer 4 Parameters 37 images Sparse connectivity.\nXXXXXXXXXXXXXXXXXXXXXX\n')
% maps = randn(15,15,150,37,'single');
% gmaps = GPUsingle(maps);
% F = randn(9,9,100,37,'single');
% gF  =GPUsingle(F);
% % C = ones(100,150,'single');
% C = single(conmat_randdoub(size(F,3),size(maps,3)));
% gC = GPUsingle(C);
% fprintf('Computing on CPU\n');
% tic;R = valid_each3_each4(maps,F,C,4);tC=toc;
% fprintf('CPU time for valid_each3_each4: %f\n',tC);
% fprintf('Computing on GPU\n');
% tic; gR = valid_each3_each4(gmaps,gF,gC,4); GPUsync; t=toc;
% fprintf('GPU time for valid_each3_each4: %f, Speedup: %f\n',t,tC/t);
% % figure, sdf(R);
% % figure, sdf(single(gR));
% fprintf('Maximum Error: %f\n\n',max(abs(R(:)-single(gR(:)))));
% % fprintf('Memory Free: %f\n',GPUmem)
% fprintf('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXx\n\n')
% size(F)
% size(gF)
% size(maps)
% size(gmaps)
% sR = size(R);
% % figure(1); clf; sdf(permute(reshape(R,[sR(1:3) prod(sR(4:5))]),[1 2 4 3]));
% % figure(2); clf; sdf(permute(reshape(single(gR),[sR(1:3) prod(sR(4:5))]),[1 2 4 3]));
% % clear/all
% 












% fprintf('XXXXXXXXXXXXXXXXXXXXXX\nLayer 1 Parameters Single\nXXXXXXXXXXXXXXXXXXXXXX\n')
% maps = randn(150,150,3,'single');
% gmaps = GPUsingle(maps);
% F = randn(7,7,3,16,'single');
% gF  =GPUsingle(F);
% C = ones(3,16,'single');
% gC = GPUsingle(C);
% fprintf('Computing on CPU\n');
% tic;
% R = full_each4_sum3(maps,F,C,4);
% tC=toc;
% fprintf('CPU time for full_each4_sum3: %f\n',tC);
% fprintf('Computing on GPU\n');
% tic; gR = full_each4_sum3(gmaps,gF,gC,4); GPUsync; t=toc;
% % figure(1), sdf(R);
% % figure(2), sdf(single(gR));
% fprintf('GPU time for full_eachJ_loopk: %f, Speedup: %f\n',t,tC/t);
% fprintf('Maximum Error: %f\n\n',max(abs(R(:)-single(gR(:)))));
% % fprintf('Memory Free: %f\n',GPUmem)
% fprintf('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXx\n\n')
% size(F)
% size(gF)
% size(maps)
% size(gmaps)
% clear all
% 
% fprintf('XXXXXXXXXXXXXXXXXXXXXX\nLayer 4 Parameters Single Image\nXXXXXXXXXXXXXXXXXXXXXX\n')
% maps = randn(7,7,50,'single');
% gmaps = GPUsingle(maps);
% F = randn(7,7,50,150,'single');
% gF  =GPUsingle(F);
% C = ones(50,150,'single');
% gC = GPUsingle(C);
% fprintf('Computing on CPU\n');
% tic; R = full_each4_sum3(maps,F,C,4);tC=toc;
% fprintf('CPU time for full_each4_sum3: %f\n',tC);
% fprintf('Computing on GPU\n');
% tic; gR = full_each4_sum3(gmaps,gF,gC,4); GPUsync; t=toc;
% fprintf('GPU time for full_eachJ_loopk: %f, Speedup: %f\n',t,tC/t);
% fprintf('Maximum Error: %f\n\n',max(abs(R(:)-single(gR(:)))));
% % fprintf('Memory Free: %f\n',GPUmem)
% fprintf('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXx\n\n')
% size(F)
% size(gF)
% size(maps)
% size(gmaps)
% clear all
% 
% fprintf('XXXXXXXXXXXXXXXXXXXXXX\nLayer 1 Parameters 37 Images, Same Filters\nXXXXXXXXXXXXXXXXXXXXXX\n')
% maps = randn(150,150,3,37,'single');
% gmaps = GPUsingle(maps);
% F = randn(7,7,3,16,'single');
% gF  =GPUsingle(F);
% C = ones(3,16,'single');
% gC = GPUsingle(C);
% fprintf('Computing on CPU\n');
% tic; R = full_each4_sum3(maps,F,C,4); tC=toc;
% fprintf('CPU time for full_each4_sum3: %f\n',tC);
% fprintf('Computing on GPU\n');
% tic; gR = full_each4_sum3(gmaps,gF,gC,4); GPUsync; t=toc;
% fprintf('GPU time for full_eachJ_loopk: %f, Speedup: %f\n',t,tC/t);
% fprintf('Maximum Error: %f\n\n',max(abs(R(:)-single(gR(:)))));
% % fprintf('Memory Free: %f\n',GPUmem)
% fprintf('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXx\n\n')
% size(F)
% size(gF)
% size(maps)
% size(gmaps)
% clear all
% 
% 
% 
% fprintf('XXXXXXXXXXXXXXXXXXXXXX\nLayer 4 Parameters 37 Images, Same Filters\nXXXXXXXXXXXXXXXXXXXXXX\n')
% maps = randn(7,7,50,37,'single');
% gmaps = GPUsingle(maps);
% F = randn(7,7,50,150,'single');
% gF  =GPUsingle(F);
% C = ones(50,150,'single');
% gC = GPUsingle(C);
% fprintf('Computing on CPU\n');
% tic; R = full_each4_sum3(maps,F,C,4); tC=toc;
% fprintf('CPU time for full_each4_sum3: %f\n',tC);
% fprintf('Computing on GPU\n');
% tic; gR = full_each4_sum3(gmaps,gF,gC,4); GPUsync; t=toc;
% fprintf('GPU time for full_eachJ_loopk: %f, Speedup: %f\n',t,tC/t);
% fprintf('Maximum Error: %f\n\n',max(abs(R(:)-single(gR(:)))));
% % fprintf('Memory Free: %f\n',GPUmem)
% fprintf('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXx\n\n')
% size(F)
% size(gF)
% size(maps)
% size(gmaps)
% clear all
% 
% fprintf('XXXXXXXXXXXXXXXXXXXXXX\nLayer 1 Parameters 37 Images, Different Filters\nXXXXXXXXXXXXXXXXXXXXXX\n')
% maps = randn(150,150,3,37,'single');
% gmaps = GPUsingle(maps);
% F = randn(7,7,3,16,37,'single');
% gF  =GPUsingle(F);
% C = ones(3,16,'single');
% gC = GPUsingle(C);
% fprintf('Computing on CPU\n');
% tic; R = full_each4_sum3(maps,F,C,4); tC=toc;
% fprintf('CPU time for full_each4_sum3: %f\n',tC);
% fprintf('Computing on GPU\n');
% tic; gR = full_each4_sum3(gmaps,gF,gC,4); GPUsync; t=toc;
% fprintf('GPU time for full_eachJ_loopk: %f, Speedup: %f\n',t,tC/t);
% fprintf('Maximum Error: %f\n\n',max(abs(R(:)-single(gR(:)))));
% % fprintf('Memory Free: %f\n',GPUmem)
% fprintf('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXx\n\n')
% size(F)
% size(gF)
% size(maps)
% size(gmaps)
% clear all
% 
% 
% 
% fprintf('XXXXXXXXXXXXXXXXXXXXXX\nLayer 4 Parameters 37 Images Different Filters\nXXXXXXXXXXXXXXXXXXXXXX\n')
% maps = randn(7,7,50,37,'single');
% gmaps = GPUsingle(maps);
% F = randn(7,7,50,150,37,'single');
% gF  =GPUsingle(F);
% C = ones(50,150,'single');
% gC = GPUsingle(C);
% fprintf('Computing on CPU\n');
% tic; R = full_each4_sum3(maps,F,C,4); tC=toc;
% fprintf('CPU time for full_each4_sum3: %f\n',tC);
% fprintf('Computing on GPU\n');
% tic; gR = full_each4_sum3(gmaps,gF,gC,4); GPUsync; t=toc;
% fprintf('GPU time for full_eachJ_loopk: %f, Speedup: %f\n',t,tC/t);
% fprintf('Maximum Error: %f\n\n',max(abs(R(:)-single(gR(:)))));
% % fprintf('Memory Free: %f\n',GPUmem)
% fprintf('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXx\n\n')
% size(F)
% size(gF)
% size(maps)
% size(gmaps)
% clear all
% 
% 
% 
% 
% 
% 
% fprintf('XXXXXXXXXXXXXXXXXXXXXX\nLayer 1 Parameters 37 Images, Same Filters, Sparse Conmats\nXXXXXXXXXXXXXXXXXXXXXX\n')
% maps = randn(150,150,3,37,'single');
% gmaps = GPUsingle(maps);
% F = randn(7,7,3,16,'single');
% gF  =GPUsingle(F);
% C = ones(3,16,'single');
% C = single(conmat_randdoub(3,16));
% % inds = randperm(numel(C));
% % inds = inds(1:floor(0.15*numel(C)));
% % C(inds) = 0
% gC = GPUsingle(C);
% fprintf('Computing on CPU\n');
% tic; R = full_each4_sum3(maps,F,C,4); tC=toc;
% fprintf('CPU time for full_each4_sum3: %f\n',tC);
% fprintf('Computing on GPU\n');
% tic; gR = full_each4_sum3(gmaps,gF,gC,4); GPUsync; t=toc;
% fprintf('GPU time for full_eachJ_loopk: %f, Speedup: %f\n',t,tC/t);
% fprintf('Maximum Error: %f\n\n',max(abs(R(:)-single(gR(:)))));
% % fprintf('Memory Free: %f\n',GPUmem)
% fprintf('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXx\n\n')
% size(F)
% size(gF)
% size(maps)
% size(gmaps)
% clear all
% 
% 
% 
% fprintf('XXXXXXXXXXXXXXXXXXXXXX\nLayer 4 Parameters 37 Images, Same Filters, Sparse Conmats\nXXXXXXXXXXXXXXXXXXXXXX\n')
% maps = randn(7,7,50,37,'single');
% gmaps = GPUsingle(maps);
% F = randn(7,7,50,150,'single');
% gF  =GPUsingle(F);
% C = ones(50,150,'single');
% C = single(conmat_randdoub(50,150));
% % inds = randperm(numel(C));
% % inds = inds(1:floor(0.15*numel(C)));
% % C(inds) = 0
% gC = GPUsingle(C);
% fprintf('Computing on CPU\n');
% tic; R = full_each4_sum3(maps,F,C,4); tC=toc;
% fprintf('CPU time for full_each4_sum3: %f\n',tC);
% fprintf('Computing on GPU\n');
% tic; gR = full_each4_sum3(gmaps,gF,gC,4); GPUsync; t=toc;
% fprintf('GPU time for full_eachJ_loopk: %f, Speedup: %f\n',t,tC/t);
% fprintf('Maximum Error: %f\n\n',max(abs(R(:)-single(gR(:)))));
% % fprintf('Memory Free: %f\n',GPUmem)
% fprintf('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXx\n\n')
% size(F)
% size(gF)
% size(maps)
% size(gmaps)
% clear all



% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% % 
% % 
% % 
% % 
% % 
% % 
% % 
% % 
% % % 
% % % 
% % % 
% % % fprintf('\n\n\n');
% % % 
% % % % Multiple images, single kernel.
% % % maps = randn(256,256,100,'single');
% % % gmaps = GPUsingle(maps);
% % % F = randn(7,7,1,'single');
% % % gF  =GPUsingle(F);
% % % 
% % % fprintf('Computing on CPU\n');
% % % tic;C = ipp_conv2(maps,F,'valid');t=toc;
% % % fprintf('CPU time for multiple image, single kernel: %f\n',t);
% % % fprintf('Computing on GPU\n');
% % % tic;G = gpu_conv2(gmaps,gF,'valid'); GPUsync; t=toc;
% % % fprintf('GPU time for multiple image, single kernel: %f\n',t);
% % % fprintf('Maximum Error: %f\n',max(abs(C(:)-single(G(:)))));
% % % 
% % % 
% % % 
% % % % Multiple kernels, single image.
% % % maps = randn(256,256,1,'single');
% % % gmaps = GPUsingle(maps);
% % % F = randn(7,7,100,'single');
% % % gF  =GPUsingle(F);
% % % 
% % % fprintf('Computing on CPU\n');
% % % tic;C = ipp_conv2(maps,F,'valid');t=toc;
% % % fprintf('CPU time for multiple kernels, single image: %f\n',t);
% % % fprintf('Computing on GPU\n');
% % % tic;G = gpu_conv2(gmaps,gF,'valid'); GPUsync; t=toc;
% % % fprintf('GPU time for multiple kernels, single image: %f\n',t);
% % % fprintf('Maximum Error: %f\n',max(abs(C(:)-single(G(:)))));
% % % 
% % % % 
% % % GPUmem
% % % Multiple kernels, Multiple images.
% % maps = randn(100,100,100,'single');
% % gmaps = GPUsingle(maps);
% % F = randn(7,7,100,'single');
% % gF  =GPUsingle(F);
% % 
% % fprintf('Computing on CPU\n');
% % tic;C = ipp_conv2(maps,F,'valid');t=toc;
% % fprintf('CPU time for multiple kernels, multiple images: %f\n',t);
% % fprintf('Computing on GPU\n');
% % tic;G = gpu_conv2(gmaps,gF,'valid'); GPUsync; t=toc;
% % fprintf('GPU time for multiple kernels, multiple images: %f\n',t);
% % size(G)
% % % selector = GPUsingle(find(eye(100,100)));
% % % G = reshape(G,size(G,1),size(G,2),size(G,3)*size(G,4));
% % % G = G(:,:,selector);
% % fprintf('Maximum Error: %f\n',max(abs(C(:)-single(G(:)))));
% % 
% 
% 
% 
% 

% clear all
% maps = randn(156,156,16,37,'single');
% tmaps = tic;
% gMaps = GPUsingle(maps);
%  GPUsync, tmapsSend = toc(tmaps)
% tmaps = tic;
% maps = single(gMaps);
%  GPUsync, tmapsReceive = toc(tmaps)
% F = randn(7,7,100,150,'single');
% tfilters = tic;
% GF = GPUsingle(F);
%  GPUsync, tfilterSend = toc(tfilters)
% tfilters=tic;
% F = single(GF);
%  GPUsync, tfilterReceive=toc(tfilters)




% 
% fprintf('XXXXXXXXXXXXXXXXXXXXXX\n3D Layer 1 Parameters 37 Images\nXXXXXXXXXXXXXXXXXXXXXX\n')
% maps = randn(256,256,40,1,4,'single');
% gmaps = GPUsingle(maps);
% F = randn(9,9,5,8,1,'single');
% gF  =GPUsingle(F);
% setSize(gF,[size(gF) 1]);
% C = ones(8,1,'single');
% gC = GPUsingle(C);
% fprintf('Computing on CPU\n');
% tic;
% R = valid_each3_sum4_3d(maps,F,C,4);
% tC=toc;
% fprintf('CPU time for valid_each3_sum4: %f\n',tC);
% fprintf('Computing on GPU\n');
% TGPU=tic; gR = valid_each3_sum4_3d(gmaps,gF,gC,4); GPUsync; t=toc(TGPU);
% % figure(1),sdf(squeeze(R));
% % figure(2), sdf(squeeze(single(gR)));
% % profile viewer
% fprintf('GPU time for valid_each3_sum4: %f, Speedup: %f\n',t,tC/t);
% fprintf('Maximum Error: %f\n',max(abs(R(:)-single(gR(:)))));
% % fprintf('Memory Free: %f\n',GPUmem)
% fprintf('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXx\n\n')
% size(F)
% size(gF)
% size(maps)
% size(gmaps)
% size(R)
% size(gR)
% clear all
% % 





