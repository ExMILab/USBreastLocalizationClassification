%This is programm for predicting the malignancy confidence of a lesion in
%breast US images;
%Paths---------------------------------------------------------------------
scripts = 'V:\Projekte\2021_Zuza_Breast_nnU-net\Revision_2\';
path = 'V:\Projekte\2021_Zuza_Breast_nnU-net\Revision_2\ECV\';
addpath(scripts); addpath(path);

%Available segmentation models---------------------------------------------
models = {'ensemble', 'fold_0','fold_1','fold_2','fold_3','fold_4'};

for i = 1:size(models,2)
    model = models{1,i};
    %First prepare the samples---------------------------------------------
    info = prepare_samples_2D(model,path);
    disp(info)
end