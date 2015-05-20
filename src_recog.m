function [ ratio ] = src_recog( dict , dict_label , test , test_label ,Tr_Num )
% This function is the SRC_BP algorithm.
% Author: Weiyang Liu  Email;wyliu@pku.edu.cn
% Input:
% dict is the input training samples.
% dict_label is the corresponding lables of training samples.
% test is the input of test samples.
% terst_label is the corresponding lables of test samples.
% Tr_num is the class number.
% Output:
% ratio is the recognition accuracy.
% If you happen to use this code, please cite:
% W. Liu, L. Lu, H. Li, W. Wang, and Y. Zou. "A Novel Kernel Collaborative Representation Approach for Image Classification." IEEE International Conference on Image Processing (ICIP), Paris, France, 2014
dict = double(dict);
test = double(test);
[ dict_ver2 ] = trdict_preprocess( dict , dict_label , Tr_Num );
%%
dict_fin = [];
dict_m = [];
for i = 1:size(dict_ver2,1)
    dict_m = cell2mat(dict_ver2(i,1));%,ones(size(cell2mat(dict(i,1))),1)*(i-1)];
    dict_fin = [dict_fin;dict_m];
end
dictionary = dict_fin;%construct the dictionary
dictionary = dictionary';
% save d.mat dictionary;
%%
dictionary = dictionary*diag(1./sqrt(sum(dictionary.*dictionary)));%normalization
sus = 0;%initilization
test = test';
test = test*diag(1./sqrt(sum(test.*test)));
test = test';
for i = 1:size(test,1)
    test2 = test(i,:);
    % figure
    % imshow(reshape(test2,28,28)');%show the original image
    sol = SolveBP(dictionary,test2',size(dict,1));%use the SolveBP to solve the sparse coefficients
    % figure
    % bar(sol)
    res = [];%initilization
    for j = 1:Tr_Num
        est_image = double(dictionary*testnew(sol,j,dict_ver2,Tr_Num));
        mm = testnew(sol,j,dict_ver2,Tr_Num);
        res(j) = double((sum((test2'-est_image).^2))/sum(mm.*mm));%calculate the residuals
    end
    % figure
    % bar(res)
    [val1,num1] = min(res);
    if num1 == test_label(i)
        sus = sus+1
    end
end
ratio = sus/i;%output the recognition rate
end
