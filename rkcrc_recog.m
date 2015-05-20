function [ ratio ] = rkcrc_recog( dict , dict_label , test , test_label , Tr_Num)
% This function is the KCRC_RLS algorithm. (No dimensionality reduction in kernel space)
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
%%
dict = double(dict);
test = double(test);
[ dict_ver2 ] = trdict_preprocess( dict , dict_label , Tr_Num );
dict_label=sort(dict_label);
%%
dict_fin = [];
dict_m = [];
for i = 1:size(dict_ver2,1)
    dict_m = cell2mat(dict_ver2(i,1));%,ones(size(cell2mat(dict(i,1))),1)*(i-1)];
    dict_fin = [dict_fin;dict_m];
end
lamda=0.001*size(dict,1)/700;
dictionary = dict_fin;%construct the dictionary
dictionary = dictionary';
% save d.mat dictionary;
dictionary = dictionary*diag(1./sqrt(sum(dictionary.*dictionary)));%normalization
%%
K=[];
for i = 1:size(dictionary,2)
    for j = 1:size(dictionary,2)
        K(i,j) = kernel(dictionary(:,i),dictionary(:,j));
    end
end
[ K2 ] = trdict_preprocess( K , dict_label ,Tr_Num );
%%
K_fin = [];
K_m = [];
for i = 1:size(K2,1)
    K_m = cell2mat(K2(i,1));%,ones(size(cell2mat(dict(i,1))),1)*(i-1)];
    K_fin = [K_fin;K_m];
end
% lamda = 0.001*size(K,1)/700;
K = K_fin;%construct the kernel dictionary
K = K';
for i =  1 : size(test,1)
    test2 = test(i,:)./norm(test(i,:));
    for b= 1:size(dictionary,2);
        test3(b,i) = kernel(dictionary(:,b),test2');
    end
    % test3(:.i) = test3(:.i)./norm(test3(:.i)); normalization
end
MaxI = 70; % iteration number
%% projection matrix computing
opts.disp = 0;
G = K'*K;
tau = eigs(double(G),1,'lm',opts);
rho = 1.25;
tau_input = tau;
mu = 1/tau;
mubar = 1e10*mu;
for i = 1:MaxI
    i;
    tau = 2/mu;
    Proj(i).M = single(inv(G+tau*eye(size(K,2)))); % way 1a
%     Proj(i).M = single(inv(G+0.001*eye(size(tr_dat,2)))); % way 1b
%     Proj(i).M = single(inv(G+tau*eye(size(tr_dat,2)))*tr_dat'); % way 2
    mu        = min(rho*mu,mubar);
end
%%
KAPPA = [2];
for ki = 1:size(KAPPA,2)
    kappa = KAPPA(ki);
% testing
ID = [];
count=0;
    for indTest = 1:size(test3,2)
        % tic
        [id] = RCRC_RLS(K,Proj,test3(:,indTest),dict_label,kappa,tau_input,MaxI-1);
        ID = [ID id];
        count = count + 1
        % toc
end
cornum = sum(ID==test_label');
ratio = [cornum/length(test_label')] % recognition rate
end