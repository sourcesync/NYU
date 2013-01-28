




% maps
maps = GPUsingle(randn(10,10,'single'));
smaps = [size(maps) 1 1];
setSize(maps,smaps);

% Pooling size.
Fs = [5 5 1 1];
F = ones(Fs,GPUsingle);
setSize(F,Fs);
% Size of the poolsed maps.
spmaps = [Fs(1) Fs(2) size(maps,1)-Fs(1)+1 size(maps,2)-Fs(2)+1 size(maps,3) size(maps,4) ];
% RESULT MAPS.
pmaps = zeros(spmaps,GPUsingle);

setSize(pmaps,spmaps);
% setSize(xinds,spmaps);
% setSize(yinds,spmaps);


px = 5;
py = 5;
sx = spmaps(3);
sy = spmaps(4);
sm = spmaps(5);
si = spmaps(6);
GPUcompileStart('gpufor_conv','-f',maps,F,pmaps,sx,sy,sm,si,Fs)
% bx = ceil(smaps(1)-Fs(1)+1);
% by = ceil(smaps(2)-Fs(2)+1);
% Get how much to add to the it,jt index to get the middle of the overlapped path.
% midx = floor(Fs(1)/2);
% midy = floor(Fs(2)/2);
% offx  =1;
% offy = 1;

GPUfor image=1:si
    GPUfor map=1:sm
        GPUfor it=1:sx
            GPUfor jt=1:sy 
%             result = 0;
                GPUfor fx=1:Fs(1)
                    GPUfor fy=1:Fs(2)
%                 GPUfor xx=1:Fs(1)
%                     GPUfor yy=1:Fs(2)
                % Linear indices starting at zero.
%                 linind = slice(pinds,it,jt,map,image);
                % y index starting at 0.
%                 yind = floor(linind/Fs(1));
%                 xind = linind - yind*Fs(1);
%                 xind = single(xind+it);
%                 yind = single(yind+jt);
% xind = slice(xinds,it,jt,map,image);
% yind = slice(yinds,it,jt,map,image);
endx = it+Fs(1)-1;
endy = jt+Fs(2)-1;
% endy = jt;
% endx = it;
% linind = slice(maps,linind-(floor(linind/Fs(1)))*Fs(1)+it,floor(linind/Fs(1))+jt,map,image);
mapsarea = slice(maps,it,jt,map,image);
% mapsarea = slice(maps,it+fx-offx,jt+fy-offy,map,image);

filter = slice(F,fx,fy,map,image);
% result = result + mapsarea*filter;
%             assign
GPUend
% Pooled maps.
GPUend
                assign(1,pmaps,mapsarea.*filter,fx,fy,it,jt,map,image);

            GPUend
        GPUend
    GPUend
GPUend
GPUcompileStop(pmaps)


