function info = make_movie(fr, images,path,name)
cd(path)

writerObj = VideoWriter(name);
writerObj.FrameRate = fr;

%secsPerImage = 0.1:0.1:(size(images,1)*0.1);

open(writerObj)

for i = 7:107%:size(images,1)
    frame = im2frame(images{i});

    writeVideo(writerObj,frame)
end

close(writerObj)

info = 'Movie was composed and saved.';
end