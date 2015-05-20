% directly run this file for a simple demo in MNIST data set.
% Author: Weiyang Liu  Email;wyliu@pku.edu.cn
% If you happen to use this code, please cite:
% W. Liu, L. Lu, H. Li, W. Wang, and Y. Zou. "A Novel Kernel Collaborative Representation Approach for Image Classification." IEEE International Conference on Image Processing (ICIP), Paris, France, 2014
%%
clear;clc;close all;
path = cd;
addpath([path '\databases\']);
addpath([path '\utilities']);
%%
% Loading MNIST database
load('train50perclassMNIST4Example.mat')
%%
% For KCRC_RLS(identity), please use
[ ratio , ~ ] = kcrc_recog( train_data , train_label , test_data , test_label , 10);
% For SRC_BP, please use 
% [ ratio ]=src_recog( train_data , train_label , test_data , test_label , 10);
% For CRC-RLS, please use
% [ ratio , ~ ]=crc_recog( train_data , train_label , test_data , test_label , 10);
% For KCRC-RLS(PCA), please use 
% [ ratio ]=kcrc_pca_recog( train_data , train_label , test_data , test_label , 10);
% For R-KCRC(identity), please use 
% [ ratio ]=rkcrc_recog( train_data , train_label , test_data , test_label , 10);