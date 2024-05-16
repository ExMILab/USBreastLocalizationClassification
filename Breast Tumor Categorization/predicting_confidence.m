%script for mixing radiomics and autoencodewr features
%also for standardizing and selecting teh final set of features
%but without labels
function info = predicting_confidence(folder, model,path)
%paths---------------------------------------------------------------------
scripts = 'V:\Projekte\2021_Zuza_Breast_nnU-net\Revision_2\';
%load radiomics signature--------------------------------------------------
fnn = strcat(path,'\',folder,'\',model,'\Radiomics_Signature_2.mat');
load(fnn)

%load classifier-----------------------------------------------------------
fnn = strcat(scripts,'\classifier_2.mat');
load(fnn)

%changing the order of features in TEST------------------------------------
p_names = SVM.RequiredVariables(1,:);
t_names = radiomics_signature.Properties.VariableNames(2:end);

TT = change_order(radiomics_signature,t_names,p_names);

%predicting confidence-----------------------------------------------------
cv_SVM = SVM.ClassificationSVM; cv_SVM = fitPosterior(cv_SVM); 
[L1,S1] = predict(cv_SVM,TT);

%confidence file-----------------------------------------------------------
confidence = filling_in_confidence_table(TT(:,1),S1(:,2),path,folder);

confidence = round(confidence.*100,1);

save(strcat(path,'\',folder,'\',model,'\Confidence_2.mat'),'confidence')

info ='Classification results saved.';
end
