function result = loopmult_m(A, B)
  
% matlab implementation
  
  [numrA, numcA, numcasesA]=size(A);
  [numrB, numcasesB] = size(B);
  assert(numcA==numrB)
  assert(numcasesA==numcasesB)

  result = zeros(numrA, numcasesA);

  for xx=1:numcasesA
    result(:, xx) = A(:, :, xx) * B(:, xx);
  end