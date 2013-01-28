function runme
%% run test


disp('* Start test: 2-D randn array');
m=16;
n=16;
A = cuRandn(m,n,GPUsingle);

A_h = single(A);
fprintf('Showing some of GPU result:\n');
A_h(1:3,1:6)
disp('* Test finished: 2-D randn array');

disp('* Start test: 3-D randn array');
m=16;
n=16;
p=32;
B = cuRandn(m,n,p,GPUsingle);

B_h = single(B);
fprintf('Showing some of GPU result:\n');
B_h(1:3,1:6,1:2)
disp('* Test finished: 3-D randn array');


%end