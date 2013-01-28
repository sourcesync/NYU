%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% A test script for variosu gpu max and min variations
% which compares speed for different types of pooling, different sizes of 
% images, and different numbers of images.
%
% @file
% @author Matthew Zeiler
% @date Apr 29, 2011
%
% @test @copybrief testMaxMin.m
% @pooling_file @copybrief testMaxMin.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
fprintf('Testing max and min over dimensions.\n');
for test=1:3
    
    switch test
        case 1
            A = randn(100,100,'single');
            fprintf('Simple Square matrix.\n');
            disp(size(A));
        case 2
            A = randn(1,1000,'single');
            fprintf('Singleton dimension.\n');
            disp(size(A));
        case 3
            A = randn(20,30,40,5,6,'single');
            fprintf('Many dimensions of different sizes\n');
            disp(size(A));
    end
    
    gA = GPUsingle(A);
    
    for i=1:ndims(A)
        fprintf('Now testing maxes dim: %d\n',i);
        tic;
        [gm,gind] =  max(gA,[],i);
        GPUsync; tg = toc;
        tic
        [m,ind] = max(A,[],i);
        tc = toc;
        fprintf('    Difference in maxes: %f\n',max(abs(single(gm(:))-m(:))));
        fprintf('    Different in inds: %f\n',max(abs(single(gind(:))-ind(:))));
        fprintf('        Speedup: %0.2f\n',tc/tg);
        fprintf('Now testing mins\n');
        tic;
        [gm,gind] =  min(gA,[],i);
        GPUsync; tg = toc;
        tic
        [m,ind] = min(A,[],i);
        tc = toc;
        fprintf('    Difference in mins: %f\n',max(abs(single(gm(:))-m(:))));
        fprintf('    Different in inds: %f\n',max(abs(single(gind(:))-ind(:))));
        fprintf('        Speedup: %0.2f\n',tc/tg);
    end
end


fprintf('\n\nTesting max and min comparisons to matrix or scalars.\n');
for test=1:4
    
    switch test
        case 1
            A = randn(1000,1000,GPUsingle);
            B = randn(1000,1000,GPUsingle);
            fprintf('Simple Square matrices.\n');
            disp(size(A));
        case 2
            A = randn(20,30,40,5,6,GPUsingle);
            B = 0.1;
            fprintf('Comparise to CPU scalar.\n');
            disp(size(A));
        case 3
            A = randn(20,30,40,5,6,GPUsingle);
            B = GPUsingle(0.1);
            fprintf('Comparse to GPU scalar\n');
            disp(size(A));
        case 4
            fprintf('Compares to self.\n');
            A = GPUsingle(randn(20,30,40,5,5,'single'));
            B = A;
    end
    
    %     gA = GPUsingle(A);
    
        tic;
        [gm] =  max(A,B);
        GPUsync; tg = toc;
        tic;
        [m] = max(single(A),single(B));
        tc =toc;
        fprintf('    Difference in maxes: %f\n',max(abs(single(gm(:))-m(:))));
        fprintf('        Speedup: %0.2f\n',tc/tg);
        
        tic;
        [gm] =  min(A,B);
        GPUsync; tg = toc;
        tic;
        [m] = min(single(A),single(B));
        tc = toc;
        fprintf('    Difference in mins: %f\n',max(abs(single(gm(:))-m(:))));
        fprintf('        Speedup: %0.2f\n',tc/tg);            
end








fprintf('Testing single argument max.\n')
for test=1:4
    
    switch test
        case 1
            A = randn(100,100,'single');
            fprintf('Simple Square matrix.\n');
            disp(size(A));
        case 2
            A = randn(1,1000,'single');
            fprintf('Singleton dimension.\n');
            disp(size(A));
        case 3
            A = randn(20,30,40,5,6,'single');
            fprintf('Many dimensions of different sizes\n');
            disp(size(A));
        case 4
            A = randn(1,1,1,45,3,'single');
            fprintf('Many singleton dimensions then a couple non singleton');
            disp(size(A));
    end
    
    gA = GPUsingle(A);
    
    
    fprintf('Now testing maxes\n');
    tic;
    [gm,gind] =  max(gA);
    GPUsync; tg = toc;
    tic
    [m,ind] = max(A);
    tc = toc;
    fprintf('    Difference in maxes: %f\n',max(abs(single(gm(:))-m(:))));
    fprintf('    Different in inds: %f\n',max(abs(single(gind(:))-ind(:))));
    fprintf('        Speedup: %0.2f\n',tc/tg);
    fprintf('Now testing mins\n');
    tic;
    [gm,gind] =  min(gA);
    GPUsync; tg = toc;
    tic
    [m,ind] = min(A);
    tc = toc;
    fprintf('    Difference in mins: %f\n',max(abs(single(gm(:))-m(:))));
    fprintf('    Different in inds: %f\n',max(abs(single(gind(:))-ind(:))));
    fprintf('        Speedup: %0.2f\n',tc/tg);
end







