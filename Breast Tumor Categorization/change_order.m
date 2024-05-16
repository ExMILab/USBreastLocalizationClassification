function TT = change_order(radiomics_signature,t_names,p_names)

%changing the order of features in TEST------------------------------------
idx_t_in_p = zeros(size(t_names,2),1);
for i = 1:size(idx_t_in_p,1)
idx_t_in_p(i,1) = find(strcmp(t_names(1,i),p_names));
end

ft = (table2array(radiomics_signature(:,2:end)));  
tmp =zeros(size(ft,1),size(ft,2));
for i = 1:size(idx_t_in_p,1)
tmp(:,idx_t_in_p(i,1)) = ft(:,i);
end

TT = radiomics_signature;
TT{:,2:end} = tmp;

tmp_names = cell(1,size(radiomics_signature,2));
tmp_names(1,2:end) = p_names;
tmp_names{1,1} = 'Label';

TT.Properties.VariableNames=tmp_names; 
end