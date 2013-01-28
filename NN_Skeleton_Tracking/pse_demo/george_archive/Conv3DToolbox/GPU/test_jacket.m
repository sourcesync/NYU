%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% A test script for various GPU jacket functions to do convolutions and pooling.
%
% @file
% @author Matthew Zeiler
% @date Apr 11, 2011
%
% @test @copybrief test_jacket.m
% @gpu_file @copybrief test_jacket.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% 
% clear all
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% fprintf('valid_eachK_loopJ\n');
% input_maps = 15;
% feature_maps = 50;
% images = 64;
% mapdim = 52;
% fdim = 7;
% z = randn(mapdim,mapdim,feature_maps,images,'single');
% F = randn(fdim,fdim,input_maps,feature_maps,'single');
% C = ones(size(F,3),size(F,4),'single');
% tic
% recon_z = valid_eachK_loopJ(z,F,C,4);
% tC = toc
% 
% 
% gz = gsingle(z);
% gF = gsingle(F);
% sz = [size(recon_z) 1 1 1 1];
% % gC = gzeros([sC(1) sC(2) sC(3) sizesC(4)]);
% grecon_z = gzeros([ sz(1) sz(2) sz(3) sz(4)]);
% gsync, tic
% % Temp variable to sum over.
% temp = gzeros([sz(1) sz(2) feature_maps]);
% 
% 
% % gfor linind=1:input_maps*images*feature_maps
% % Split up over whole batch and 
% gfor linind=1:input_maps*images
%     [jt,im] = ind2sub([input_maps images],linind);
% % Sum over this dimension.
% for kt=1:feature_maps    
%     % [jt,im,kt] = ind2sub([input_maps images feature_maps],linind);
%     % linind = jt + (im-1)*input_maps;
%     % linind2 = sub2ind([input_maps images],jt,im);
% %     A = ;
% %     B = ;
% %     grecon_z(:,:,linind,kt) = conv2(gz(:,:,kt,im),gF(:,:,jt,kt),'valid'); 
%     temp(:,:,kt) = conv2(gz(:,:,kt,im),gF(:,:,jt,kt),'valid'); 
% 
% end
% grecon_z(:,:,jt,im) = sum(temp,3);
% gend
% % grecon_z = sum(grecon_z,4);
% % grecon_z = reshape(grecon_z,[sz(1) sz(2) sz(3) sz(4)]);
% geval(grecon_z)
% gsync, tG=toc
% 
% 
% max(abs(va(single(grecon_z)))-abs(recon_z(:)))
% fprintf('valid_eachK_loopJ speedup: %0.2fx\n',tC/tG);
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% 
% 
% clear all
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% fprintf('full_eachK_loopJ\n');
% input_maps = 100;
% feature_maps = 150;
% images = 64;
% mapdim = 9;
% fdim = 7;
% ims = randn(mapdim,mapdim,input_maps,images,'single');
% F = randn(fdim,fdim,input_maps,feature_maps,'single');
% C = ones(size(F,3),size(F,4),'single');
% tic
% recon_z = full_eachJ_loopK(ims,F,C,4);
% tC = toc
% 
% 
% gims = gsingle(ims);
% gF = gsingle(F);
% sz = [size(recon_z) 1 1 1 1];
% % gC = gzeros([sC(1) sC(2) sC(3) sizesC(4)]);
% grecon_z = zeros([ sz(1) sz(2) feature_maps*images input_maps],gsingle);
% gsync, tic
% 
% 
% % gfor linind=1:input_maps*images*feature_maps
% % Split up over whole batch and 
% gfor linind2=1:input_maps*images
%     % Temp variable to sum over.
% % temp = gzeros([sz(1) sz(2) input_maps]);
% 
%     [jt,im] = ind2sub([input_maps images],linind2);
% % Sum over this dimension.
% for kt=1:feature_maps    
%     % [jt,im,kt] = ind2sub([input_maps images feature_maps],linind);
% 
%     linind = sub2ind([feature_maps images],kt,im);
%     % linind = jt + (im-1)*input_maps;
%     % linind2 = sub2ind([input_maps images],jt,im);
% %     grecon_z(:,:,linind,kt) = conv2(gz(:,:,kt,im),gF(:,:,jt,kt),'valid'); 
%     grecon_z(:,:,linind,jt) = conv2(gims(:,:,jt,im),gF(:,:,jt,kt),'full'); 
% 
% end
% gend
% % figure(2), sdf(single(temp))
% % keyboard
% grecon_z = sum(grecon_z,4);
% grecon_z = reshape(grecon_z,[sz(1) sz(2) feature_maps images]);
% geval(grecon_z)
% gsync, tG=toc
% size(grecon_z)
% 
% max(abs(va(single(grecon_z)))-abs(recon_z(:)))
% fprintf('full_eachK_loopJ speedup: %0.2fx\n',tC/tG);
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% 
% clear all
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% fprintf('valid_loop_loop\n');
% input_maps = 3;
% feature_maps = 15;
% images = 64;
% mapdim = 156;
% imsdim = 150;
% z = randn(mapdim,mapdim,feature_maps,images,'single');
% ims = randn(imsdim,imsdim,input_maps,images,'single');
% C = ones(size(ims,3),size(z,3),'single');
% tic
% recon_z = valid_loopK_loopJ(z(end:-1:1,end:-1:1,:,:),ims,C,4);
% tC = toc
% 
% gz = gsingle(z);
% gims = gsingle(ims);
% sC = [size(recon_z) 1 1 1];
% grecon_z = gzeros([sC(1:2) sC(3)*sC(4) sC(5)]);
% im=1;
% gsync, tic
% gfor linind=1:input_maps*feature_maps*images
% % gfor im=1:size(A,4)
% % Imput maps.
% % for jt=1:size(gims,3)
%     % Feature maps
% %     for kt=1:size(gz,3)
% [jt,kt,im] = ind2sub([input_maps feature_maps images],linind);
% %     size(conv2(gA(:,:,it),gB(:,:,jt,it),'valid'))
% grecon_z(:,:,(kt-1)*input_maps+jt,im) = conv2(flipdim(flipdim(gz(:,:,kt,im),1),2),gims(:,:,jt,im),'valid');
% %     end
% % end
% gend
% % grecon_z = reshape(grecon_z,[sC(1) sC(2) input_maps feature_maps images]);
% geval(grecon_z)
% gsync, tG=toc
% 
% 
% max(abs(single(grecon_z(:)))-abs(recon_z(:)))
% fprintf('valid_loopK_loopJ speedup %0.2fx\n',tC/tG);
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% clear all
% 

%
% fprintf('Max Pooling\n');
%
% A = randn(156,156,2,1,'single');
% tic
% for im=1:size(A,3)
% bA(:,:,im) = im2col(A(:,:,im),[3 3],'distinct');
% end
% tC = toc
%
% tic
% pA = max_pool(A,[3 3],[],2);
% tP=toc
%
% gA = gsingle(A);
% gsync, tic
% for im=1:size(gA,3)
%     if(im==1)
%         gbA = gzeros(size(bA));
%     end
% gbA(:,:,im) = im2col(gA(:,:,im),[3 3],'distinct');
% end
% gbA = max(gbA,[],1);
% gbA = reshape(gbA,[sqrt(size(gbA,2)) sqrt(size(gbA,2)) size(gbA,3)]);
% gsync, tG=toc
%
%
% max(abs(single(gbA(:)))-abs(pA(:)))
%
%
%
%


ckernel_fft_rot = randn(100,100,2,2,'single');
cvis_fft = randn(100,100,2,'single');
cC = zeros(100,100,2,'single');

kernel_fft_rot = GPUsingle(ckernel_fft_rot);
vis_fft = GPUsingle(cvis_fft);
C = GPUsingle(cC);
tic
  for i=1:size(ckernel_fft_rot,4)
    for j=1:size(cvis_fft,3)
      out=real(ifft2( cvis_fft(:,:,j).*ckernel_fft_rot(:,:,j,i) ));
      cC(:,:,i)= cC(:,:,i) + out(1:end,1:end);
    end
  end
tc = toc


tic
forloop1(kernel_fft_rot,vis_fft,C);
GPUsync, tg=toc
