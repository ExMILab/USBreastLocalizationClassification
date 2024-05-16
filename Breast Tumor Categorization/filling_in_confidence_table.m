function confidence = filling_in_confidence_table(s_names,pp,path,folder)
%--------------------------------------------------------------------------
scripts = 'V:\Projekte\2021_Zuza_Breast_nnU-net\Revision_2\';
addpath(scripts);
%--------------------------------------------------------------------------
path_img = strcat(path,'\',folder,'\images\image_after_process\Renamed\');
cd(path_img); images = struct2cell(dir('*.jpg'))'; 
all_names = images(:,1);
%--------------------------------------------------------------------------
s_names = table2cell(s_names);

idx = reducing_feature_space(s_names',all_names');

confidence = zeros(size(images,1),1);

for i = 1:size(pp,1)
    confidence(idx(i,1),1) = pp(i,1);
end
end
