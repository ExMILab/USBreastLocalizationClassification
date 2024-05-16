function info = corpp_image(img, r_bb, name, path)

if isempty(r_bb)==1
    info = 'Image and mask for autoenocder features extrcation was not saved. No object loacted in the frame';
else

    img_cr = imresize(imcrop(img,r_bb),[128,128]);
    mask = imbinarize(ones(128,128));

    fnn1 = strcat(path,'\Autoencoder\images\',name);
    imwrite(img_cr,fnn1)

    fnn2 = strcat(path,'\Autoencoder\masks\',name);
    imwrite(mask,fnn2)

    info = 'Image and mask for autoenocder features extrcation was saved';
end
end