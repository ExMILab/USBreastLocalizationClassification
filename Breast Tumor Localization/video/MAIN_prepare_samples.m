%This is programm for predicting the malignancy confidence of a lesion in
%breast US images/movies;

%Paths---------------------------------------------------------------------
scripts = 'V:\Projekte\2021_Zuza_Breast_nnU-net\Revision_2\';
path = 'V:\Projekte\2021_Zuza_Breast_nnU-net\Revision_2\SEGMENTATION_VIDEOS\';
%path = 'V:\Projekte\2021_Zuza_Breast_nnU-net\Revision_2\';
addpath(scripts); addpath(path);

%Available folders---------------------------------------------------------
% folders = {'002_010_20230807',...
%     '002_011_20230814',...
%     '002_012_20230731',...
%     '002_012_20230814',...
%     '002_013_20230818',...
%     '002_013_20230901',...
%     '002_014_20230907',...
%     '002_014_20230928'};

folders = {'002_019_20240206','002_020_20240206'};

%Available segmentation models---------------------------------------------
models = {'ensemble', 'fold_0','fold_1','fold_2','fold_3','fold_4',...
'ensemble_1','fold_01','fold_11','fold_21','fold_31','fold_41',...
'ensemble_2','fold_02','fold_12','fold_22','fold_32','fold_42'};

for i = 1:size(folders,2)
    %info = rename_frame(folders{1,i},models,path);
    %disp(info)
for ii = 1:size(models,2)
    folder = folders{1,i}; model = models{1,ii};
    %First prepare the samples---------------------------------------------
    info = prepare_samples(folder, model,path);
    disp(info)
end
end
