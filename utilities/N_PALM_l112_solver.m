% =========================================================================
%   Lei Zhang, Meng Yang, and Xiangchu Feng,
%   "Sparse Representation or Collaborative Representation: Which Helps Face
%    Recognition?" in ICCV 2011.
%
%   Collaborative Representation based Classification for Face Recognition
%   Lei Zhang, Meng Yang, Xiangchu Feng, Yi Ma, David Zhang, ARXIV: http://arxiv.org/abs/1204.2358v2
%
% Written by Meng Yang @ COMP HK-PolyU
% July, 2011 & March, 2014
% =========================================================================
function [x,e,ER] = N_PALM_l112_solver(b,A,pinv_D,kappa,tau,MaxI)
% rou>1
rho = 1.25 ;
[m,n] = size(A) ;

% At = A';
% G = At*A ;
% opts.disp = 0;
% tau = eigs(G,1,'lm',opts);
mu = 1/tau ;
mubar = 1e10*mu ;

lambda = ones(m,1) ;
x = zeros(n,1);
e = zeros(m,1);
tol = 1e-6 ;
converged_main = false;
nIter = 0;
f = [x;e];
ER = [];
while ~converged_main & nIter<=MaxI
    nIter = nIter + 1 ;
    
%     x = inv((mu/2)*A'*A+eye(size(A,2)))*A'*tem_y*mu/2;
%     x = inv(A'*A+eye(size(A,2))*2/mu)*A'*tem_y;

    tem_y = A'*(b-e+1/mu*lambda); x = pinv_D(nIter).M*tem_y; % way 1
%     x     = pinv_D(nIter).M*(b-e+1/mu*lambda);   % way 2
    
%     tem_y = (b-e+1/mu*lambda);
%     x = pinv_D(nIter).M*tem_y;

    tt_y  = b-A*x+1/mu*lambda;
    e = shrink(tt_y, kappa/mu);
    
%     lambda = lambda + mu*(b - A*x-e) ;
    lambda = mu*(tt_y-e);
    mu = min(rho*mu,mubar) ;
%     plot([x;e],'-o');title(num2str(nIter));pause(1);
    
    prev_f = f;
    f = [x;e];
    criterionObjective = norm(f-prev_f,2)/norm(f,2);
    ER = [ER criterionObjective];
    converged_main =  ~(criterionObjective > tol);
end
ER = ER(3:end);

function Y = shrink(X, alpha)
    
Y = sign(X).*max(abs(X)-alpha,0) ;