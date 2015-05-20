% dict_fin=[];dict_m=[];
% for i=1:size(dict,1)
%     dict_m=cell2mat(dict(i,1));%,ones(size(cell2mat(dict(i,1))),1)*(i-1)];
%     dict_fin=[dict_fin;dict_m];
% end
% dictionary=dict_fin;%¹¹½¨×Öµä
% % save d.mat dictionary;
% dictionary=dictionary';
% imshow(uint8(reshape(dictionary(:,3),28,28)'))
function [ out ] = testnew( x , n , dict , Tr_Num )
% Setting the coefficients for the other classes as 0s.
% dict_num=0;
dict_num(1) = 0;
for i = 1:Tr_Num
    dict_num(i+1) = size(cell2mat(dict(i,1)),1);
end
for i = 1:Tr_Num
    if i ~= n
        for j = 1:dict_num(i+1)
            x(sum(dict_num(1:i))+j) = 0;
        end
    end
end
out = x;
end