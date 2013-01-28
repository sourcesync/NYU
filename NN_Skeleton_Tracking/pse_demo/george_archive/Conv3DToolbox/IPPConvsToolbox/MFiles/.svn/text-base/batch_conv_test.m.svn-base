%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% Tests that doing multiple single image convolutions gives the same result
% as a single batch convolution.
%
% @file
% @author Matthew Zeiler
% @date Dec 5, 2010
%
% @test @copybrief batch_conv_test.m
% @ipp_file @copybrief batch_conv_test.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


clear all

% fprintf('XXXXXXXXXXXXXXXXXXXXXX\nLayer 1 Parameters 10 Images Recon Down\nXXXXXXXXXXXXXXXXXXXXXX\n')
% maps = rand(166,166,16,1,'single');
% F = rand(7,7,3,16,'single');
% C = ones(3,16,'single');
% fprintf('Computing on CPU as full batch\n');
% tCPU = tic;
% B = valid_eachK_loopJ(maps,F,C);
% tC=toc(tCPU);
% fprintf('CPU time for valid_eachK_loopJ: %f\n',tC);
% fprintf('Computing on CPU 2as full batch\n');
% % F2 = repmat(F,[1 1 1 1 size(maps,4)]);
% F2 = F;
% tCPU2 = tic;
% B2 = squeeze(sum(valid_eachK_loopJ_ipp2(maps,F2,C),4));
% tC2=toc(tCPU2);
% fprintf('CPU2 time for valid_eachK_loopJ: %f\n',tC2);
% fprintf('Computing on CPU as separate images\n');
% S = zeros(size(B),'single');
% tGPU=tic; 
% for i=1:size(maps,4)
%     S(:,:,:,i) = valid_eachK_loopJ(maps(:,:,:,i),F2(:,:,:,:,i),C);
% end
% t=toc(tGPU);
% fprintf('CPU time for valid_eachK_loopJ separately: %f, Batch Speedup: %f, CPU2 Speedup %f \n',t,t/tC,tC/tC2);
% fprintf('Maximum Error: %f\n',max(abs(B(:)-single(S(:)))));
% fprintf('Maximum Error: %f\n',max(abs(B2(:)-single(S(:)))));
% fprintf('Some elements of Separate then batch:\n')
% disp(squeeze(cat(2,S(1:3,1:3,2,:),B(1:3,1:3,2,:))));
% disp(squeeze(cat(2,B(1:3,1:3,2,:),B2(1:3,1:3,2,:))));
% fprintf('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXx\n\n')
% clear all


% fprintf('XXXXXXXXXXXXXXXXXXXXXX\nLayer 1 Parameters 10 Images Recon Down Multiple Filters\nXXXXXXXXXXXXXXXXXXXXXX\n')
% maps = rand(166,166,16,10,'single');
% F = rand(7,7,3,16,10,'single');
% C = ones(3,16,'single');
% fprintf('Computing on CPU as full batch\n');
% tCPU = tic;
% B = valid_eachK_loopJ(maps,F,C);
% tC=toc(tCPU);
% fprintf('CPU time for valid_eachK_loopJ: %f\n',tC);
% fprintf('Computing on CPU as separate images\n');
% S = zeros(size(B),'single');
% tGPU=tic; 
% for i=1:10
%     S(:,:,:,i) = valid_eachK_loopJ(maps(:,:,:,i),F(:,:,:,:,i),C);
% end
% t=toc(tGPU);
% fprintf('CPU time for valid_eachK_loopJ separately: %f, Batch Speedup: %f\n',t,t/tC);
% fprintf('Maximum Error: %f\n',max(abs(B(:)-single(S(:)))));
% fprintf('Some elements of Separate then batch:\n')
% disp(squeeze(cat(2,S(1:3,1:3,2,:),B(1:3,1:3,2,:))));
% fprintf('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXx\n\n')
% clear all


% 
% 
% fprintf('XXXXXXXXXXXXXXXXXXXXXX\nLayer 1 Parameters 10 Images Forward Prop\nXXXXXXXXXXXXXXXXXXXXXX\n')
% maps = rand(166,166,3,10,'single');
% F = rand(7,7,3,16,'single');
% C = ones(3,16,'single');
% fprintf('Computing on CPU as full batch\n');
% tCPU = tic;
% B = full_eachJ_loopK(maps,F,C);
% tC=toc(tCPU);
% fprintf('CPU time for full_eachJ_loopK: %f\n',tC);
% fprintf('Computing on CPU as separate images\n');
% S = zeros(size(B),'single');
% tGPU=tic; 
% for i=1:10
%     S(:,:,:,i) = full_eachJ_loopK(maps(:,:,:,i),F,C);
% end
% t=toc(tGPU);
% fprintf('CPU time for full_eachJ_loopK separately: %f, Batch Speedup: %f\n',t,t/tC);
% fprintf('Maximum Error: %f\n',max(abs(B(:)-single(S(:)))));
% fprintf('Some elements of Separate then batch:\n')
% disp(squeeze(cat(2,S(1:3,1:3,2,:),B(1:3,1:3,2,:))));
% 
% fprintf('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXx\n\n')
% clear all
% 

% fprintf('XXXXXXXXXXXXXXXXXXXXXX\nLayer 1 Parameters 10 Images Forward Prop Mult Filters\nXXXXXXXXXXXXXXXXXXXXXX\n')
% maps = rand(156,156,15,'single');
% ims = rand(150,150,1,'single');
% F = rand(7,7,1,15,'single');
% C = ones(1,15,'single');
% fprintf('Computing on CPU as full batch\n');
% tCPU = tic;
% B = full_eachJ_loopK(ims,F,C);
% tC=toc(tCPU);
% fprintf('CPU1 time for full_eachJ_loopK: %f\n',tC);
% fprintf('Computing on CPU with on IPP as full batch\n');
% tCPU2 = tic;
% B2 = squeeze(sum(full_eachJ_loopK_ipp(ims,F,C),3));
% tC2=toc(tCPU2);
% fprintf('Computing on CPU with on IPP old as full batch\n');
% tCPU3 = tic;
% B3 = squeeze(sum(full_eachJ_loopK_old(ims,F,C),3));
% tC3=toc(tCPU3);
% fprintf('Maximum Error with IPP New versus IPP Old: %.15f\n',max(abs(B2(:)-single(B3(:)))));
% fprintf('Maximum Error with IPP New versus conv2: %.15f\n',max(abs(B2(:)-single(B(:)))));
% fprintf('Maximum Error with IPP Old versus conv2: %.15f\n',max(abs(B3(:)-single(B(:)))));
% 



% 
% fprintf('XXXXXXXXXXXXXXXXXXXXXX\nLayer 1 Parameters 10 Images Forward Prop Mult Filters\nXXXXXXXXXXXXXXXXXXXXXX\n')
% % F = rand(7,7,1,15,'single');
% % C = ones(1,15,'single');
% fprintf('Computing on CPU as full batch\n');
% tCPU = tic;
% B = squeeze(sum(valid_eachK_loopJ(maps,F,C),4));
% tC=toc(tCPU);
% fprintf('CPU1 time for full_eachJ_loopK: %f\n',tC);
% fprintf('Computing on CPU with on IPP as full batch\n');
% tCPU2 = tic;
% B2 = squeeze(sum(valid_eachK_loopJ_ipp(maps,F,C),4));
% tC2=toc(tCPU2);
% fprintf('Computing on CPU with on IPP old as full batch\n');
% tCPU3 = tic;
% B3 = squeeze(sum(valid_eachK_loopJ_old(maps,F,C),4));
% tC3=toc(tCPU3);
% fprintf('Maximum Error with IPP New versus IPP Old: %.15f\n',max(abs(B2(:)-single(B3(:)))));
% fprintf('Maximum Error with IPP New versus conv2: %.15f\n',max(abs(B2(:)-single(B(:)))));
% fprintf('Maximum Error with IPP Old versus conv2: %.15f\n',max(abs(B3(:)-single(B(:)))));





% 
% 
% fprintf('XXXXXXXXXXXXXXXXXXXXXX\nLayer 1 Parameters 10 Images Forward Prop Mult Filters\nXXXXXXXXXXXXXXXXXXXXXX\n')
% % F = rand(7,7,1,15,'single');
% % C = ones(1,15,'single');
% fprintf('Computing on CPU as full batch\n');
% tCPU = tic;
% B = valid_loopK_loopJ(maps,ims,C) ;
% tC=toc(tCPU);
% fprintf('CPU1 time for full_eachJ_loopK: %f\n',tC);
% fprintf('Computing on CPU with on IPP as full batch\n');
% tCPU2 = tic;
% B2 = valid_loopK_loopJ_ipp(maps,ims,C);
% tC2=toc(tCPU2);
% fprintf('Computing on CPU with on IPP old as full batch\n');
% tCPU3 = tic;
% B3 = valid_loopK_loopJ_old(maps,ims,C);
% tC3=toc(tCPU3);
% fprintf('Maximum Error with IPP New versus IPP Old: %.15f\n',max(abs(B2(:)-single(B3(:)))));
% fprintf('Maximum Error with IPP New versus conv2: %.15f\n',max(abs(B2(:)-single(B(:)))));
% fprintf('Maximum Error with IPP Old versus conv2: %.15f\n',max(abs(B3(:)-single(B(:)))));
% 


% B2 = zeros(size(B),'single');
% tCPU2=tic; 
% for i=1:10
%     B2(:,:,:,i) = squeeze(sum(full_eachJ_loopK_ipp2(maps(:,:,:,i),F(:,:,:,:,i),C),3));
% end
% tC2=toc(tCPU2);
% fprintf('CPU2 time for full_eachJ_loopK: %f\n',tC2);
% fprintf('Computing on CPU as separate images\n');
% S = zeros(size(B),'single');
% tGPU=tic; 
% for i=1:size(F,5)
%     S(:,:,:,i) = full_eachJ_loopK(maps(:,:,:,i),F(:,:,:,:,i),C);
% end
% t=toc(tGPU);
% fprintf('CPU time for full_eachJ_loopK separately: %f, Batch Speedup: %f CPU2 Speedup over CPU1 %f\n',t,t/tC,tC/tC2);
% fprintf('Maximum Error with cpu1: %f\n',(abs(B(:)-single(B2(:)))));

% fprintf('Maximum Error with cpu2: %f\n',max(abs(B2(:)-single(S(:)))));
% fprintf('Some elements of Separate then batch:\n')
% disp(squeeze(cat(2,S(1:3,1:3,2,:),B(1:3,1:3,2,:))));
% disp(squeeze(cat(2,B(1:3,1:3,2,:),B2(1:3,1:3,2,:))));
% fprintf('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXx\n\n')
% clear all










% 
% fprintf('XXXXXXXXXXXXXXXXXXXXXX\nLayer 1 Parameters 10 Images Forward Prop\nXXXXXXXXXXXXXXXXXXXXXX\n')
% maps = rand(106,106,16,1,'single');
% F = rand(100,100,3,1,'single');
% C = ones(3,16,'single');
% fprintf('Computing on CPU as full batch\n');
% tCPU = tic;
% B = valid_loopK_loopJ(maps,F,C);
% keyboard
% B2 = valid_loopK_loopJ(maps,F,C);
% keyboard
% B3 = valid_loopK_loopJ_ipp(maps,F,C);
% keyboard
% tC=toc(tCPU);
% fprintf('CPU time for valid_loopK_loopJ: %f\n',tC);
% fprintf('Computing on CPU as separate images\n');
% S = zeros(size(B),'single');
% tGPU=tic; 
% for i=1:10
%     S(:,:,:,:,i) = valid_loopK_loopJ(maps(:,:,:,i),F(:,:,:,i),C);
% end
% t=toc(tGPU);
% fprintf('CPU time for valid_loopK_loopJ separately: %f, Batch Speedup: %f\n',t,t/tC);
% fprintf('Maximum Error: %f\n',max(abs(B(:)-single(S(:)))));
% fprintf('Some elements of Separate then batch:\n')
% disp(squeeze(cat(2,S(1:3,1:3,2,4,:),B(1:3,1:3,2,4,:))));
% fprintf('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXx\n\n')
% clear all


% fprintf('XXXXXXXXXXXXXXXXXXXXXX\nLayer 1 Parameters 10 Images Forward Prop\nXXXXXXXXXXXXXXXXXXXXXX\n')
% maps = rand(106,106,16,10,'single');
% F = rand(100,100,3,10,'single');
% C = ones(3,16,'single');
% fprintf('Computing on CPU as full batch\n');
% tCPU = tic;
% B = valid_loopK_loopJ(maps,F,C);
% tC=toc(tCPU);
% fprintf('CPU time for valid_loopK_loopJ: %f\n',tC);
% fprintf('Computing on CPU as separate images\n');
% S = zeros(size(B),'single');
% tGPU=tic; 
% for i=1:10
%     S(:,:,:,:,i) = valid_loopK_loopJ(maps(:,:,:,i),F(:,:,:,i),C);
% end
% t=toc(tGPU);
% fprintf('CPU time for valid_loopK_loopJ separately: %f, Batch Speedup: %f\n',t,t/tC);
% fprintf('Maximum Error: %f\n',max(abs(B(:)-single(S(:)))));
% fprintf('Some elements of Separate then batch:\n')
% disp(squeeze(cat(2,S(1:3,1:3,2,4,:),B(1:3,1:3,2,4,:))));
% fprintf('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXx\n\n')
















