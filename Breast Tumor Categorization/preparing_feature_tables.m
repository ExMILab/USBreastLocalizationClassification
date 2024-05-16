%script for mixing radiomics and autoencodewr features
%also for standardizing and selecting teh final set of features
%but without labels

function info = preparing_feature_tables(folder, model,path)
%paths---------------------------------------------------------------------
scripts = 'V:\Projekte\2021_Zuza_Breast_nnU-net\Revision_2\';
path_workspace = strcat(path,'\',folder,'\',model);
path_rad = strcat(path_workspace,'\Radiomics\');
path_ae = strcat(path_workspace,'\Autoencoder\');

%compose feature tables----------------------------------------------------
rad_features = rad_features_table(path_rad);
ae_features = ae_features_table(path_ae);

%combine feature tables----------------------------------------------------
table_mix = [ae_features(:,1) rad_features(:,2:end) ae_features(:,2:end)];

%standardize feature tables------------------------------------------------
load(strcat(scripts,'standardize_2.mat'))

tmp = table_mix{:,2:end}; 
tmp_s = (tmp-ms)./rs;
table_mix{:,2:end} = tmp_s;

%select features-----------------------------------------------------------
load(strcat(scripts,'select_2.mat'))

rs_names = ['Image' rs_names];

%rs_names{1,1} ='Image';
all_names = table_mix.Properties.VariableNames; 

idx_names1 = reducing_feature_space(rs_names,all_names);

radiomics_signature = table_mix(:,idx_names1);

fnn = strcat(path_workspace,'\Radiomics_Signature_2.mat');
save(fnn,'radiomics_signature')

info = 'Feature tables prepared.';
end