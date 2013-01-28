%CUSAMPLEMULTINOMIAL Sample from many same-size multinomial distributions
%in parallel
%
%   CUSAMPLEMULTINOMIAL(A,r,B) Samples from each multinomial (columns of
%   A) and stores the resulting binary vectors in the columns of B)
%   Note that if the columns of A do not quite sum to 1, the remaining
%   amount is the probability that all elements (in a column) are "off"
%   i.e. the multinomial is actually over K+1 possibilities, where K is
%   the number of rows of A
%   INPUTS: 
%   A : K x numMulti multinomials
%   B : K x numMulti binary matrix result
%   r : numMulti x 1 uniformly random samples
