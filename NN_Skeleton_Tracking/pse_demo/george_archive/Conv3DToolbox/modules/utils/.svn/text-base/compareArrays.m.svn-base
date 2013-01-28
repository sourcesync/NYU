function compareArrays(h_C_ref, h_C, epsilon)
% compareArrays Numerically compare two arrays

global GPUtest

try
  type = GPUtest.type;
catch
  error('GPUtest not initialized. Please use GPUtestInit.');
end


%%h_C_ref = h_C_ref(1:end);
%%h_C = h_C(1:end);

if (numel(h_C_ref)~=numel(h_C))
  GPUtestLOG('Number of elements is different',1);
  return;
end

% check if both empty
if (isempty(h_C_ref) && isempty(h_C))
  return
end

% compared array has only zero elements. Generates error or just warning
% depending on GPUtest.noCompareZeros
if (nnz(h_C_ref)==0)
  GPUtestLOG('Warning: Compared arrays are zeros',GPUtest.noCompareZeros);
end

if (nnz(h_C)==0)
  GPUtestLOG('Warning: Compared arrays are zeros',GPUtest.noCompareZeros);
end


idx_nonzero = find(abs(h_C_ref));
idx_zero = find(~abs(h_C_ref));

ref_error_nonzero = 0;
ref_error_zero = 0;

if (~isempty(idx_nonzero))
  ref_error_nonzero = max(abs((h_C_ref(idx_nonzero) - h_C(idx_nonzero))./h_C_ref(idx_nonzero)));
end
if (~isempty(idx_zero))
  ref_error_zero = max(abs(h_C_ref(idx_zero) - h_C(idx_zero)));
end

ref_error = max(ref_error_nonzero,ref_error_zero);
if (ref_error < epsilon)
else
  GPUtestLOG (['Arrays are different (error is ' num2str(ref_error) ')!!!'], 1);
end


%GPUtestLOG(' ',0);

end