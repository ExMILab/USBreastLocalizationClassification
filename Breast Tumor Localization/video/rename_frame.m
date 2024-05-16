function info = rename_frame(folder,models,path)
for ii = 1:size(models,2)
    %paths---------------------------------------------------------------------
    path_img = strcat(path,'\',folder,'\images\image_after_process');
    path_seg = strcat(path,'\',folder,'\',models{1,ii},'\Segments\');
    
    %for images----------------------------------------------------------------
    cd(path_img)
    images = dir('*.jpg');
    
    mkdir Renamed\
    %--------------------------------------------------------------------------
    for i = 1:size(images,1)
        img = imread(strcat(path_img,'\frame_',num2str(i),'.jpg'));
    
        if i < 10
            fnn = strcat(path_img,'\Renamed\frame_00', num2str(i),'.jpg');
            imwrite(img,fnn)
    
        elseif i < 100
            fnn = strcat(path_img,'\Renamed\frame_0', num2str(i),'.jpg');
            imwrite(img,fnn)
        else
            fnn = strcat(path_img,'\Renamed\frame_', num2str(i),'.jpg');
            imwrite(img,fnn)
    
        end
    
    end
    %for segments--------------------------------------------------------------
    cd(path_seg)
    masks = dir('*.png');
    
    mkdir Renamed\
    %--------------------------------------------------------------------------
    for i = 1:size(masks,1)
        img = imread(strcat(path_seg,'\frame_',num2str(i),'.png'));
    
        if i < 10
            fnn = strcat(path_seg,'\Renamed\frame_00', num2str(i),'.jpg');
            imwrite(img,fnn)
    
        elseif i < 100
            fnn = strcat(path_seg,'\Renamed\frame_0', num2str(i),'.jpg');
            imwrite(img,fnn)
        else
            fnn = strcat(path_seg,'\Renamed\frame_', num2str(i),'.jpg');
            imwrite(img,fnn)
    
        end
    
    end
%--------------------------------------------------------------------------
info = 'Frames renamed.';
end
end



