function [r_seg,ss] = refine_segment(seg,area)

stats = regionprops(seg,'Area'); a = cell2mat(struct2cell(stats));
if isempty(a)==1
    r_seg = seg;
    ss = 0;
else

    idx = a == max(a); bg = a(1,sort(find(idx==1)));
    
    if size(bg,2)>1
        bg = bg(1,1);
    end

    seg = bwareaopen(seg,bg-1);

    ss = sum(sum(seg)); pp = 1.1*ss;

    while ss < pp
        seg = imfill(seg,'holes');
        seg = bwmorph(seg,'diag',inf); 
        seg = bwmorph(seg,'bridge',inf);
        seg = bwmorph(seg,'thicken');
        seg = imfill(seg,'holes');
            
        ss = sum(sum(seg));
    end

    if area > 0
        if ss > area
            r_seg = seg;
        else
            r_seg = imbinarize(zeros(size(seg,1),size(seg,1)));
        end
    else
        r_seg = seg;
    end

end