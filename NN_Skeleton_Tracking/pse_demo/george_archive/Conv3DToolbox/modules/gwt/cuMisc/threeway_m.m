function prodABC = threeway_m(A,B,C)
  
%compute three-way product
%prod(aa,bb,cc) sum_xx A(aa,xx)*B(bb,xx)*C(cc,xx)
  
  [numA,ncA]=size(A);
  [numB,ncB]=size(B);
  [numC,ncC]=size(C);
  assert(ncA==ncB)
  assert(ncB==ncC);
  
  prodABC = zeros(numA,numB,numC);
  
  for xx=1:ncA
    prodABC = prodABC + ...
              reshape(kron(C(:,xx),kron(B(:,xx),A(:,xx))), ...
                      [numA numB numC]);
  end