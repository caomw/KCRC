function [ out ] = kernel( a,b )
% This is the kernel function. 
% We use RBF kernel here.
mu=0.5;
out=exp(-mu*(norm((a-b),2))^2);
end

