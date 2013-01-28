%CUMATRIXTOGRID reverse the operation performed by CUGRIDTOMATRIX (like
%col2im but in parallel)
%
%   CUMATRIXTOGRID(A,B,f) takes (f*f) blocks (stored as the columns of B)
%   and arranges them into square images, stored as the columns of A
%
%   INPUTS: 
%   A :   (f*f) x regionsPerImage*numCases SQUARE blocks
%   B : (imgSize * imgSize) x numCases resulting SQUARE images 
%   f : Region size (in each of x and y) 2<= p <= 16
%
%   NOTE: since CUGRIDTOMATRIX transposes each (f*f) block before writing
%     we assume here that blocks are transposed