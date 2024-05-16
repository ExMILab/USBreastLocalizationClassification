function idx = reducing_feature_space(names1,names2)
%this function finds names from names1 in names2 and retursn the position
%of the finding as vector
%remmeber that vectors need to have the same shape to be compared

idx = zeros(size(names1,2),1);

for i = 1:size(idx,1)
    a = find(strcmp(names2,names1(1,i)));
    
    if isempty(a)== 1 
        idx(i,1) = 0;
    else
        idx(i,1) = a;
    end
end

%idx_names1 = sort(idx_names1);
end