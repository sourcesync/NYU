function targets = samplemultinomial_cpu(multi,r)
  
%Columns of multi are multinomials of size "nomials" + 1
%Returns a matrix where at most 1 element per col is on sampled according
%to the distribution given by each column
  [nomials,multinomials]=size(multi);
  
  targets = zeros(size(multi));
  
  for ii=1:multinomials
    s = 0; prevSum = 0;
    rnd = r(ii);
    for xx=1:nomials
      s = s + multi(xx,ii);
      targets(xx,ii)= (rnd>prevSum && rnd < s);
      prevSum = s;
    end
  end
  
  