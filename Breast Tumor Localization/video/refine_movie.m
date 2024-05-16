function r_segments = refine_movie(segments)

i = 3;

while i < size(segments,1)-1
    
%     figure(1)
%     subplot(1,5,1), imshow(segments{i-2})
%     subplot(1,5,2), imshow(segments{i-1})
%     subplot(1,5,3), imshow(segments{i})
%     subplot(1,5,4), imshow(segments{i+1})
%     subplot(1,5,5), imshow(segments{i+2})
    
    seg = segments{i,1}; seg_b = segments{i-1,1}; seg_a = segments{i+1,1};
        
    seg_bb = segments{i-2,1};  seg_aa = segments{i+2,1};
    
    if sum(sum(seg_bb))==0 && sum(sum(seg_b))>0 && sum(sum(seg)) == 0 && sum(sum(seg_a))>0 && sum(sum(seg_aa))== 0
        segments{i-1,1} = imbinarize(zeros(size(seg,1),size(seg,2)));
        segments{i+1,1} = imbinarize(zeros(size(seg,1),size(seg,2)));
        
    elseif sum(sum(seg_bb))>0 && sum(sum(seg_b))>0 && sum(sum(seg)) == 0 && sum(sum(seg_a))>0 && sum(sum(seg_aa))== 0
        segments{i,1} = seg_a + seg_b;
        
    elseif sum(sum(seg_bb))==0 && sum(sum(seg_b))>0 && sum(sum(seg)) == 0 && sum(sum(seg_a))>0 && sum(sum(seg_aa))>0
        segments{i,1} = seg_a + seg_b;
    
    elseif sum(sum(seg_bb))>0 && sum(sum(seg_b))>0 && sum(sum(seg)) == 0 && sum(sum(seg_a))>0 && sum(sum(seg_aa))>0
        segments{i,1} = seg_a + seg_b;
    
    elseif sum(sum(seg_bb))==0 && sum(sum(seg_b))==0 && sum(sum(seg)) > 0 && sum(sum(seg_a))==0 && sum(sum(seg_aa))==0
        segments{i,1} = imbinarize(zeros(size(seg,1),size(seg,2)));
        
    elseif sum(sum(seg_bb))>0 && sum(sum(seg_b))==0 && sum(sum(seg))>0 && sum(sum(seg_a))==0 && sum(sum(seg_aa))== 0
        segments{i,1} = imbinarize(zeros(size(seg,1),size(seg,2)));
        
    elseif sum(sum(seg_bb))==0 && sum(sum(seg_b))>0 && sum(sum(seg))>0 && sum(sum(seg_a))==0 && sum(sum(seg_aa))== 0
        segments{i,1} = imbinarize(zeros(size(seg,1),size(seg,2)));
        segments{i-1,1} = imbinarize(zeros(size(seg,1),size(seg,2)));
    
    elseif sum(sum(seg_bb))==0 && sum(sum(seg_b))==0 && sum(sum(seg))>0 && sum(sum(seg_a))>0 && sum(sum(seg_aa))== 0
        segments{i,1} = imbinarize(zeros(size(seg,1),size(seg,2)));
        segments{i+1,1} = imbinarize(zeros(size(seg,1),size(seg,2)));
        
    elseif sum(sum(seg_bb))==0 && sum(sum(seg_b))==0 && sum(sum(seg))>0 && sum(sum(seg_a))==0 && sum(sum(seg_aa))>0
        segments{i,1} = imbinarize(zeros(size(seg,1),size(seg,2)));
        
    elseif sum(sum(seg_bb))>0 && sum(sum(seg_b))==0 && sum(sum(seg))>0 && sum(sum(seg_a))==0 && sum(sum(seg_aa))>0
         segments{i-2,1} = imbinarize(zeros(size(seg,1),size(seg,2)));
         segments{i,1} = imbinarize(zeros(size(seg,1),size(seg,2)));
         segments{i+2,1} = imbinarize(zeros(size(seg,1),size(seg,2)));
            
        
    end
    
%     subplot(1,5,1), imshow(segments{i-2})
%     subplot(1,5,2), imshow(segments{i-1})
%     subplot(1,5,3), imshow(segments{i})
%     subplot(1,5,4), imshow(segments{i+1})
%     subplot(1,5,5), imshow(segments{i+2})

    i = i+1;
end

r_segments = segments;

%check if there is segment in the frame
%if yes then fine
%if not then check adjacent frames
%if segment is present in both then make a average

end