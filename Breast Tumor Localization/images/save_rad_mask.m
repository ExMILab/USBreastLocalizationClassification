function info = save_rad_mask(img, r_bb, name, path)

if isempty(r_bb)==1
   info = 'Image and mask for autoenocder features extrcation was not saved. No object loacted in the frame';
else
    %make a mask of bbox and save it
    tmp = zeros(size(img,1),size(img,2));

    tmp(r_bb(2):r_bb(2)+r_bb(4)-1,r_bb(1):r_bb(1)+r_bb(3)-1) = 1;

    tmp = imbinarize(tmp);

    fnn1 = strcat(path,'\Radiomics\images\',name);
    imwrite(img,fnn1)

    fnn2 = strcat(path,'\Radiomics\masks\',name);
    imwrite(tmp,fnn2)

    info = 'Image and mask for autoenocder features extrcation was saved';
end

end