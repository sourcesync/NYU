%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%>
% This script makes the common videos that train_recon_phase_all saves out
% (saved as png files for each frame).
%
% @file
% @author Matthew Zeiler
% @date Nov 17, 2010
%
%
% @plotting_file @copybrief make_videos.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% By default get the files directly from the save path.
% path = model.fullsavepath;
% phases = model.num_phases;

path = '/misc/FergusGroup/zeiler/Results/trainAll/101_150_max_new_30s_train1/CN_gray_15_50_100_150/Run_3/reconAll/101_150_max_new_30s_train1/Run_20'
phases = 4;

% Change this to the fps it takes for each phase.
infpses = [1 1 1 1 1 1 1 1 1 1];


if(strcmp(path(end),'/')==0)
    path = [path '/'];
end

path = [path 'reconAll/'];



% Where to copy the videos to. 
outpath = '/misc/FergusGroup/zeiler/cvpr2011figs/supp/sample2421_20/'
mkdir([outpath]);

for ph=1:phases
    
    
    phpath = [path '/video_phase' num2str(ph) '/'];
    

    fprintf('\n\n  Making image reconstruction video for phase %d \n\n',ph);
    make_video(phpath,sprintf('recon%d_',ph),sprintf('recon%d',ph),[],infpses(ph),30);
    copyfile(sprintf('%srecon%d.mp4',phpath,ph),sprintf('%srecon%d.mp4',outpath,ph));
    
%     fprintf('\n\n  Making filter video for phase %d \n\n',ph);
%     make_video(phpath,sprintf('filt%d_',ph),sprintf('filt%d',ph),[],infpses(ph),30);
%     copyfile(sprintf('%sfilt%d.mp4',phpath,ph),sprintf('%sfilt%d.mp4',outpath,ph));    
    
    fprintf('\n\n  Making pixel filter video for phase %d \n\n',ph);
    make_video(phpath,sprintf('pfilt%d_',ph),sprintf('pfilt%d',ph),[],infpses(ph),30);
    if(ph>1)
    copyfile(sprintf('%spfilt%d.mp4',phpath,ph),sprintf('%spfilt%d.mp4',outpath,ph));    
    end
    fprintf('\n\n  Making feature map video for phase %d \n\n',ph);
    make_video(phpath,sprintf('feat%d_',ph),sprintf('feat%d',ph),[],infpses(ph),30);
    copyfile(sprintf('%sfeat%d.mp4',phpath,ph),sprintf('%sfeat%d.mp4',outpath,ph));    
    
    fprintf('\n\n  Making feature histogram video for phase %d \n\n',ph);
    make_video(phpath,sprintf('feathist%d_',ph),sprintf('feathist%d',ph),[],infpses(ph),30);
    copyfile(sprintf('%sfeathist%d.mp4',phpath,ph),sprintf('%sfeathist%d.mp4',outpath,ph));
        
end





























