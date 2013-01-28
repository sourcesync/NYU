function makeinstall(filesfilters, inpath, outpath)

disp('**** MAKE INSTALL ****');

% check for outpath. Create if necessary
if (~exist(outpath,'dir'))
  disp(['mkdir -> '  outpath ]);
  mkdir(outpath);
end

%% copy files
for i=1:length(filesfilters)
  filesfilter = filesfilters{i};
  targetfiles = fullfile(inpath, filesfilter);
  file = dir(targetfiles);
  
  
  for kk=1:length(file)
    % do not copy . and ..
    if strcmp(file(kk).name,'.')||strcmp(file(kk).name,'..')
    else
      infile = fullfile(inpath,file(kk).name );
      outfile = fullfile(outpath,file(kk).name );
      disp(['copy ' infile ' -> ' outfile ]);
      if (ispc)
        copyfile(infile,outfile);
      else
        system(['cp -r ' infile  ' ' outfile]);
      end
    end
  end
  
end


end