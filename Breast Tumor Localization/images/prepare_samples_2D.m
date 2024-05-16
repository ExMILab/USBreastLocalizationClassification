function info = prepare_samples_2D(model,path)

%paths---------------------------------------------------------------------
path_img = strcat(path,'\images\image_after_process\');
path_mask = strcat(path,'\',model,'\Segments\');
path_workspace = strcat(path,'\',model);

%make folder---------------------------------------------------------------
cd(path_workspace);
mkdir Segments_Refined\

mkdir Autoencoder\
mkdir Autoencoder\images\
mkdir Autoencoder\masks\
mkdir Autoencoder\results\

mkdir Radiomics\
mkdir Radiomics\images\
mkdir Radiomics\masks\
mkdir Radiomics\features\

%estimate no. of frames----------------------------------------------------
cd(path_img); 
frames = dir('*.jpg'); 
nf = size(dir('*.jpg'),1);
cd(path_mask); 
masks = dir('*.png');

%set variables-------------------------------------------------------------
images = cell(nf,1); 
areas = cell(nf,1); 
segments = cell(nf,1); 
bboxes = cell(nf,1);

%Segments refining commence here-------------------------------------------
%First: estimste the median area of the computed masks---------------------
for i = 1:nf
    frame = imbinarize(imread(strcat(masks(1).folder,'\',masks(i).name)));
    area = 0;

    [r_seg,ss] = refine_segment(frame,area);
    
    segments{i,1} = r_seg; areas{i,1} = ss;
end

r_segments = segments;

%Fourth: perform last refinement of masks------------------------------------
for i = 1:nf
    img = rgb2gray(imread(strcat(frames(1).folder,'\',frames(i).name)));
    frame = r_segments{i,1};
        
    [r_seg,~] = refine_segment(frame,areas{i,1});
        
    images{i,1} = img; r_segments{i,1} = r_seg;
        
    fnn = strcat(path_workspace,'\Segments_Refined\',frames(i).name);
    
    imwrite(r_seg,fnn)
end

%Last: compute Bounding boxes; prepare samples for feature extrcation both
%radiomics and autoencoder
for i = 1:nf
    img = images{i,1}; frame = r_segments{i,1};
            
    r_bb = extract_bbox(frame); bboxes{i,1} = r_bb;
    

    info = save_rad_mask(img, r_bb, frames(i).name, path_workspace);
    disp(info)

    info = corpp_image(img, r_bb, frames(i).name, path_workspace);
    disp(info)
end


%saving--------------------------------------------------------------------
fnn1 = strcat(path_workspace,'\Images.mat');
save(fnn1,'images');
fnn2 = strcat(path_workspace,'\Segments.mat');
save(fnn2,'r_segments')
fnn3 = strcat(path_workspace,'\BBoxes.mat');
save(fnn3,'bboxes')


info ='Samples preparation: completed';
end