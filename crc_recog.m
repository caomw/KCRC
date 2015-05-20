function [ ratio , predict_label ] = crc_recog( dict , dict_label , test , test_label , Tr_Num)
% This function is the CRC_RLS algorithm.
% Author: Weiyang Liu  Email;wyliu@pku.edu.cn
% Input:
% dict is the input training samples.
% dict_label is the corresponding lables of training samples.
% test is the input of test samples.
% terst_label is the corresponding lables of test samples.
% Tr_num is the class number.
% Output:
% ratio is the recognition accuracy.
% predict_label is the recognition results for each test samples.
% If you happen to use this code, please cite:
% W. Liu, L. Lu, H. Li, W. Wang, and Y. Zou. "A Novel Kernel Collaborative Representation Approach for Image Classification." IEEE International Conference on Image Processing (ICIP), Paris, France, 2014
dict = double(dict);
test = double(test);
[ dict_ver2 ] = trdict_preprocess( dict , dict_label , Tr_Num );
dict_label = sort(dict_label);
%%
dict_fin = [];
dict_m = [];
for i = 1:size(dict_ver2,1)
    dict_m = cell2mat(dict_ver2(i,1));%,ones(size(cell2mat(dict(i,1))),1)*(i-1)];
    dict_fin = [dict_fin;dict_m];
end
lamda = 0.001*size(dict,1)/700;
dictionary = dict_fin;%construct the dictionary
dictionary = dictionary';
% save d.mat dictionary;
%%
dictionary = dictionary*diag(1./sqrt(sum(dictionary.*dictionary)));%normalization
sus = 0;%initilization
predict_label = [];
% po=1;
invmat = inv(dictionary'*dictionary+lamda*eye(size(dict,1)));
test = test';
test = test*diag(1./sqrt(sum(test.*test)));
test = test';
for i = 1:size(test,1)
    test2 = test(i,:);%./norm(test(i,:));
    % imshow(reshape(test2,28,28)');%show the original image
    % q=dictionary*dictionary';
    sol = invmat*dictionary'*test2';%calculate the sparse coefficients
    % figure
    % bar(sol)
    res = [];%initilization
    for j = 1:Tr_Num
        est_image = double(dictionary*testnew(sol,j,dict_ver2,Tr_Num));
        mm = testnew(sol,j,dict_ver2,Tr_Num);
        res(j) = double((sum((test2'-est_image).^2))/sum(mm.*mm));%calculate the residuals
    end
    % po=po+1;
    % figure
    % bar(res)
    [val1,num1] = min(res);
    predict_label = [predict_label,num1];
    if num1 == test_label(i)
        sus = sus+1
    end
end
% TotalNum(i)=num1;
ratio = sus/i;%output the recognition rate
end
