%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% Tests the relative speed of different convolution methods, including
% ipp_conv2.
%
% @file
% @author Matthew Zeiler
% @date Mar 11, 2010
%
% @ipp_file @copybrief conv_test.m
% @test @copybrief conv_test.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear;
%%%%%% Run pack before you execute this script
%%% script to test speed of different convolution algorithms
NUM_CYCLES = 1;
BIG_IM_SIZE = 2000;
SMALL_KERNEL_SIZE = 5;
BIG_KERNEL_SIZE = 30;
SMALL_IMAGE_SIZE = 32;
NUM_SMALL_IMAGE = 10000;
MODE = 'valid';
%MODE = 'full';

fprintf(1,'\n1. Large grayscale 2D image - %dx%d pixels, kernel size = %dx%d\n',BIG_IM_SIZE,BIG_IM_SIZE,SMALL_KERNEL_SIZE,SMALL_KERNEL_SIZE);

for i=1:NUM_CYCLES
  
  a = single(rand(BIG_IM_SIZE,BIG_IM_SIZE));
  b = single(rand(SMALL_KERNEL_SIZE,SMALL_KERNEL_SIZE));
  
  tic; tmp = conv2(a,b,MODE); t1(i) = toc;
    
  clear a b tmp
    
  a = single(rand(BIG_IM_SIZE,BIG_IM_SIZE));
  b = single(rand(SMALL_KERNEL_SIZE,SMALL_KERNEL_SIZE));
 
  tic; tmp = ipp_conv2(a,b,MODE); t2(i) = toc;

end
    
fprintf(1,'Results --- Matlab (conv2): %3.2f secs. IPP: %3.2f secs. Speedup: %2.1f\n',mean(t1),mean(t2),mean(t1/t2));

%%%%%%%%%%%%%%%%%%%%%%

fprintf(1,'\n2. Large grayscale 2D image - %dx%d pixels, kernel size = %dx%d\n',BIG_IM_SIZE,BIG_IM_SIZE,BIG_KERNEL_SIZE,BIG_KERNEL_SIZE);

for i=1:NUM_CYCLES
  
  a = single(rand(BIG_IM_SIZE,BIG_IM_SIZE));
  b = single(rand(BIG_KERNEL_SIZE,BIG_KERNEL_SIZE));
  
  tic; tmp = real(ifft2(fft2(a).*fft2(b,BIG_IM_SIZE,BIG_IM_SIZE))); t1(i) = toc;
    
  clear a b tmp
    
  a = single(rand(BIG_IM_SIZE,BIG_IM_SIZE));
  b = single(rand(BIG_KERNEL_SIZE,BIG_KERNEL_SIZE));
   
  tic; tmp = ipp_conv2(a,b,MODE); t2(i) = toc;

end
    
fprintf(1,'Results --- Matlab (fft2): %3.2f secs. IPP: %3.2f secs. Speedup: %2.1f\n',mean(t1),mean(t2),mean(t1/t2));

%%%%%%%%%%%%%%%%%%%%%%

fprintf(1,'\n3. Large color 2D image - %dx%dx3 pixels, kernel size = %dx%d\n',BIG_IM_SIZE,BIG_IM_SIZE,SMALL_KERNEL_SIZE,SMALL_KERNEL_SIZE);

for i=1:NUM_CYCLES

  a = single(rand(BIG_IM_SIZE,BIG_IM_SIZE,3));
  b = single(rand(SMALL_KERNEL_SIZE,SMALL_KERNEL_SIZE));
  tmp = single(zeros(BIG_IM_SIZE,BIG_IM_SIZE,3));

  tic; for j=1:3, tmp(:,:,j) = real(ifft2(fft2(a(:,:,j)).*fft2(b,BIG_IM_SIZE,BIG_IM_SIZE))); end; t1(i) = toc;
    size(tmp)
  clear a b tmp
    
  a = single(rand(BIG_IM_SIZE,BIG_IM_SIZE,3));
  b = single(rand(SMALL_KERNEL_SIZE,SMALL_KERNEL_SIZE));
    
  tic; tmp = ipp_conv2(a,b,MODE); t2(i) = toc;
size(tmp)
end
    
fprintf(1,'Results --- Matlab (conv2): %3.2f secs. IPP: %3.2f secs. Speedup: %2.1f\n',mean(t1),mean(t2),mean(t1/t2));


%%%%%%%%%%%%%%%%%%%%%%

fprintf(1,'\n4. Large color 2D image - %dx%dx3 pixels, kernel size = %dx%d\n',BIG_IM_SIZE,BIG_IM_SIZE,BIG_KERNEL_SIZE,BIG_KERNEL_SIZE);

for i=1:NUM_CYCLES

  a = single(rand(BIG_IM_SIZE,BIG_IM_SIZE,3));
  b = single(rand(BIG_KERNEL_SIZE,BIG_KERNEL_SIZE));
  tmp = single(zeros(BIG_IM_SIZE,BIG_IM_SIZE,3));
  
  tic; for j=1:3, tmp(:,:,j) = real(ifft2(fft2(a(:,:,j)).*fft2(b,BIG_IM_SIZE,BIG_IM_SIZE))); end; t1(i) = toc;
    
  clear a b tmp
    
  a = single(rand(BIG_IM_SIZE,BIG_IM_SIZE,3));
  b = single(rand(BIG_KERNEL_SIZE,BIG_KERNEL_SIZE));
    
  tic; tmp = ipp_conv2(a,b,MODE); t2(i) = toc;

end
    
fprintf(1,'Results --- Matlab (fft2): %3.2f secs. IPP: %3.2f secs. Speedup: %2.1f\n',mean(t1),mean(t2),mean(t1/t2));


%%%%%%%%%%%%%%%%%%%%%%

fprintf(1,'\n5. Multiple tiny images - %d x %d x %d, kernel size = %dx%d\n',SMALL_IMAGE_SIZE,SMALL_IMAGE_SIZE,NUM_SMALL_IMAGE,SMALL_KERNEL_SIZE,SMALL_KERNEL_SIZE);

for i=1:NUM_CYCLES

  a = single(rand(SMALL_IMAGE_SIZE,SMALL_IMAGE_SIZE,NUM_SMALL_IMAGE));
  b = single(rand(SMALL_KERNEL_SIZE,SMALL_KERNEL_SIZE));
  if strcmp(MODE,'valid')
    tmp = single(zeros(SMALL_IMAGE_SIZE-SMALL_KERNEL_SIZE+1,SMALL_IMAGE_SIZE-SMALL_KERNEL_SIZE+1,NUM_SMALL_IMAGE));
  else
    tmp = single(zeros(SMALL_IMAGE_SIZE+SMALL_KERNEL_SIZE-1,SMALL_IMAGE_SIZE+SMALL_KERNEL_SIZE-1,NUM_SMALL_IMAGE));
  end
  
  tic; for j=1:NUM_SMALL_IMAGE, tmp(:,:,j) = conv2(a(:,:,j),b,MODE); end; t1(i) = toc;
    
  clear a b tmp
    
  a = single(rand(SMALL_IMAGE_SIZE,SMALL_IMAGE_SIZE,NUM_SMALL_IMAGE));
  b = single(rand(SMALL_KERNEL_SIZE,SMALL_KERNEL_SIZE));
    
  tic; tmp = ipp_conv2(a,b,MODE); t2(i) = toc;

end
    
fprintf(1,'Results --- Matlab (conv2): %3.2f secs. IPP: %3.2f secs. Speedup: %2.1f\n',mean(t1),mean(t2),mean(t1/t2));

%%%%%%%%%%%%%%%%%%%%%%

fprintf(1,'\n5. Multiple tiny images and kernels - %d x %d x %d, kernel size = %dx%dx%d\n',SMALL_IMAGE_SIZE,SMALL_IMAGE_SIZE,NUM_SMALL_IMAGE,SMALL_KERNEL_SIZE,SMALL_KERNEL_SIZE,NUM_SMALL_IMAGE);

for i=1:NUM_CYCLES

  a = single(rand(SMALL_IMAGE_SIZE,SMALL_IMAGE_SIZE,NUM_SMALL_IMAGE));
  b = single(rand(SMALL_KERNEL_SIZE,SMALL_KERNEL_SIZE,NUM_SMALL_IMAGE));
  if strcmp(MODE,'valid')
    tmp = single(zeros(SMALL_IMAGE_SIZE-SMALL_KERNEL_SIZE+1,SMALL_IMAGE_SIZE-SMALL_KERNEL_SIZE+1,NUM_SMALL_IMAGE));
  else
    tmp = single(zeros(SMALL_IMAGE_SIZE+SMALL_KERNEL_SIZE-1,SMALL_IMAGE_SIZE+SMALL_KERNEL_SIZE-1,NUM_SMALL_IMAGE));
  end
  
  tic; for j=1:NUM_SMALL_IMAGE, tmp(:,:,j) = conv2(a(:,:,j),b(:,:,j),MODE); end; t1(i) = toc;
    
  clear a b tmp
    
  a = single(rand(SMALL_IMAGE_SIZE,SMALL_IMAGE_SIZE,NUM_SMALL_IMAGE));
  b = single(rand(SMALL_KERNEL_SIZE,SMALL_KERNEL_SIZE,NUM_SMALL_IMAGE));
    
  tic; tmp = ipp_conv2(a,b,MODE); t2(i) = toc;

end
    
fprintf(1,'Results --- Matlab (conv2): %3.2f secs. IPP: %3.2f secs. Speedup: %2.1f\n',mean(t1),mean(t2),mean(t1/t2));

%%%%% 4 threads on horatio.cs.nyu.edu (two dual core Xeons, i.e. 4 cores)
% $$$ >> conv_test
% $$$ 
% $$$ 1. Large grayscale 2D image - 2000x2000 pixels, kernel size = 5x5
% $$$ Results --- Matlab (conv2): 0.7 secs. IPP: 0.1 secs. Speedup: 7.2
% $$$ 
% $$$ 2. Large grayscale 2D image - 2000x2000 pixels, kernel size = 30x30
% $$$ Results --- Matlab (fft2): 1.3 secs. IPP: 0.3 secs. Speedup: 4.5
% $$$ 
% $$$ 3. Large color 2D image - 2000x2000x3 pixels, kernel size = 5x5
% $$$ Results --- Matlab (conv2): 3.9 secs. IPP: 0.1 secs. Speedup: 26.6
% $$$ 
% $$$ 4. Large color 2D image - 2000x2000x3 pixels, kernel size = 30x30
% $$$ Results --- Matlab (fft2): 3.9 secs. IPP: 0.3 secs. Speedup: 11.6
% $$$ 
% $$$ 5. Multiple tiny images - 32 x 32 x 10000, kernel size = 5x5
% $$$ Results --- Matlab (conv2): 1.8 secs. IPP: 0.1 secs. Speedup: 33.1

% 8 threads on django (four dual core Xeons)
% $$$ 
% $$$ 1. Large grayscale 2D image - 2000x2000 pixels, kernel size = 5x5
% $$$ Results --- Matlab (conv2): 0.62 secs. IPP: 0.12 secs. Speedup: 5.3
% $$$ 
% $$$ 2. Large grayscale 2D image - 2000x2000 pixels, kernel size = 30x30
% $$$ Results --- Matlab (fft2): 1.17 secs. IPP: 0.26 secs. Speedup: 4.4
% $$$ 
% $$$ 3. Large color 2D image - 2000x2000x3 pixels, kernel size = 5x5
% $$$ Results --- Matlab (conv2): 3.54 secs. IPP: 0.13 secs. Speedup: 26.6
% $$$ 
% $$$ 4. Large color 2D image - 2000x2000x3 pixels, kernel size = 30x30
% $$$ Results --- Matlab (fft2): 3.59 secs. IPP: 0.31 secs. Speedup: 11.5
% $$$ 
% $$$ 5. Multiple tiny images - 32 x 32 x 10000, kernel size = 5x5
% $$$ Results --- Matlab (conv2): 1.69 secs. IPP: 0.03 secs. Speedup: 62.6
