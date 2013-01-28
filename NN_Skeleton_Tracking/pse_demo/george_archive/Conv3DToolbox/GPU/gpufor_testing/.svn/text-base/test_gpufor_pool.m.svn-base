










% maps
maps = GPUsingle(randn(10,10,'single'));
smaps = [size(maps) 1 1];
setSize(maps,smaps);

% Pooling size.
ps = [3 3];
% Size of the poolsed maps.
spmaps = [ceil(size(maps,1)/ps(1)) ceil(size(maps,2)/ps(2)) size(maps,3) size(maps,4)];
% Pooled indices.
pinds = ones(spmaps,GPUsingle);
% yinds = ones(spmaps,GPUsingle);
% Pooled maps.
pmaps = zeros(spmaps,GPUsingle);
setSize(pinds,spmaps);
setSize(pmaps,spmaps);
% setSize(xinds,spmaps);
% setSize(yinds,spmaps);


px = 5;
py = 5;
GPUcompileStart('gpufor_pool_w_indices','-f',maps,pinds,pmaps,smaps,ps)
bx = ceil(smaps(1)/ps(1));
by = ceil(smaps(2)/ps(2));
GPUfor image=1:smaps(4)
    GPUfor map=1:smaps(3)
        GPUfor it=1:bx
            GPUfor jt=1:by    
%                 GPUfor xx=1:ps(1)
%                     GPUfor yy=1:ps(2)
                % Linear indices starting at zero.
                linind = slice(pinds,it,jt,map,image);
                % y index starting at 0.
%                 yind = floor(linind/ps(1));
%                 xind = linind - yind*ps(1);
%                 xind = single(xind+it);
%                 yind = single(yind+jt);
% xind = slice(xinds,it,jt,map,image);
% yind = slice(yinds,it,jt,map,image);
% linind = slice(maps,linind-(floor(linind/ps(1)))*ps(1)+it,floor(linind/ps(1))+jt,map,image);

                assign(1,pmaps,slice(maps,linind,jt,map,image),it,jt,map,image);
%             assign
% GPUend
% GPUend
            GPUend
        GPUend
    GPUend
GPUend
GPUcompileStop(pmaps)


