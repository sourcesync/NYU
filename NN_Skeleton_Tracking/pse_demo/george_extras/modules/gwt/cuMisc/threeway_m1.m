function prodABC = threeway_m1(A,B,C)
  
%compute three-way product
%prod(aa,bb,cc) sum_xx A(aa,xx)*B(bb,xx)*C(cc,xx)
  
  [numA,ncA]=size(A);
  [numB,ncB]=size(B);
  [numC,ncC]=size(C);
  assert(ncA==ncB)
  assert(ncB==ncC);
  
  prodABC = zeros(numA,numB,numC);

  for aa=1:numA
    for bb=1:numB
      for cc=1:numC
        prodABC(aa,bb,cc) = sum( A(aa,:).*B(bb,:).*C(cc,:),2);
      end
    end
  end
  
  
  