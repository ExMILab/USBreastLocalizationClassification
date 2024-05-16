function info = make_movie_confidence(folder, model,path)
%load confidence-----------------------------------------------------------
path_results = strcat(path,'\',folder,'\',model,'\Results');
path_img = strcat(path,'\',folder,'\images\image_after_process\');

cd(strcat(path,'\',folder,'\',model))
mkdir Results 

load BBoxes.mat
load Confidence_2.mat

nf = size(bboxes,1);combined_t_img = cell(nf,1); combined_img=cell(nf,1);
%--------------------------------------------------------------------------
for i = 7:107%nf
    img = imread(strcat(path_img,'frame_',num2str(i),'.jpg')); 
    r_bb = bboxes{i,1};
            
    if isempty(r_bb)==1
        combined = img;
        combined_t = img;
    else
        if confidence(i) < 23
            text_str = ['Probably Benign Finding (BI-RADS 3): ' num2str(confidence(i)) '%'];
            combined = insertShape(rgb2gray(img),"FilledRectangle",r_bb,'Color',[120 94 240],'Opacity',0.3);
            combined_t = insertText(combined,[0 0],text_str,'FontSize',20,'BoxColor',[120 94 240],'BoxOpacity',0.9);
        elseif confidence(i) < 95
            text_str = ['Suggestive of Abnormality (BI-RADS 4): ' num2str(confidence(i)) '%'];
            combined = insertShape(rgb2gray(img),"FilledRectangle",r_bb,'Color',[220 38 127],'Opacity',0.3);
            combined_t = insertText(combined,[0 0],text_str,'FontSize',20,'BoxColor',[220 38 127],'BoxOpacity',0.9);
        elseif confidence(i) >= 95
            text_str = ['Highly Suggestive of Malignancy (BI-RADS 5): ' num2str(confidence(i)) '%'];
            combined = insertShape(rgb2gray(img),"FilledRectangle",r_bb,'Color',[254 97 0],'Opacity',0.3);
            combined_t = insertText(combined,[0 0],text_str,'FontSize',20,'BoxColor',[254 97 0],'BoxOpacity',0.9);
        end
    end

    combined_img{i,1} = combined;
    combined_t_img{i,1} = combined_t;
end

%fr = 2;
%v_name = 'combined_images_fr_2_rs2.avi';
%info = make_movie(fr,combined_t_img,path_results,v_name);
%disp(info)

%fr = 5;
%v_name = 'combined_images_fr_5_rs2.avi';
%info = make_movie(fr,combined_t_img,path_results,v_name);
%disp(info)

fr = 10;
%v_name = 'combined_images_fr_10_rs2.avi';
v_name = 'Supplemental_Video_Malignant_Lesion.avi';
info = make_movie(fr,combined_t_img,path_results,v_name);
disp(info)

fnn1 = strcat(path_results,'\Combined_without_text_rs2.mat');
save(fnn1,'combined_img');
fnn2 = strcat(path_results,'\Combined_with_text_rs2.mat');
save(fnn2,'combined_t_img')

end