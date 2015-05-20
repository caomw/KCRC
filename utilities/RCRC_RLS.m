function [id]= RCRC_RLS(D,Proj,y,Dlabels,kappa,tau_input,MaxI)
% CRC_RLS classification function
[coef,residual,ER] = N_PALM_l112_solver(y,D,Proj,kappa,tau_input,MaxI);
for ci = 1:max(Dlabels)
    coef_c   =  coef(Dlabels==ci);
    Dc       =  D(:,Dlabels==ci);
    error(ci) = norm(y-Dc*coef_c-residual,2)^2/sum(coef_c.*coef_c);
end

index      =  find(error==min(error));
id         =  index(1);