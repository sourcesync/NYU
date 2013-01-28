function GPUtestLOG(text, iserror)
% GPUtestLOG writes to the log file
% GPUtestLOG writes to the log file, or generates an error if the flag
% iserror = 1

global GPUtest

try
  type = GPUtest.type;
catch
  error('GPUtest not initialized. Please use GPUtestInit.');
end

% if printdisp = 1 writes also to standard output
printdisp = GPUtest.printDisplay;

if GPUtest.stopOnError==1
  if (iserror)
    error(text);
  end
end

fid = fopen(GPUtest.logFile,'a+');

[ST ,I] = dbstack(1);
if (iserror)
  for i=1:length(ST)
    str = sprintf('*** Error in file %s, line %d\n', ST(i).file, num2str(ST(i).line));
    fprintf(fid,str);
    if printdisp
      disp(str);
    end
  end
  str = sprintf('*** Error %s\n',text);
  fprintf(fid,str);
  if printdisp
    disp(str);
  end
else
  fprintf(fid,'%s\n',text);
  if printdisp
    disp(text);
  end
end
fclose(fid);


end


