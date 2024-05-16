function r_bb = extract_bbox(r_seg)

stats = regionprops(r_seg,'BoundingBox'); a = struct2cell(stats);

if isempty(a)==1
    r_bb = [];
else
    r_bb = cell2mat(a);

end
end