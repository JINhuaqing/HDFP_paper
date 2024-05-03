function [wild2, TT, cutoff] = wild2(alpha,n,N,S)
% This returns the output for
%  testing
%% input variables
% input 1: alpha is the significance level, often set alpha=0.05
% input 2: n is the sample size
% input 3: N is the resample size, e.g, N=10000
% input 4: S is the cell(1,n) quantity, with each S{i}=n^{-1/2}*\hat{S}_i,
%          where \hat{S}_i is defined on page 19 of the paper, calculated
%          from \hat{\eta} and \hat{w} from penalized regression and dantizig respectively.
%% output variable
% output 1: wild2=1 if reject the null H0: beta_j=0 for j\in Hn
%                =0 if accept the null
%%
 cutoff=wild1(alpha,n,N,S);
 T=cell2mat(S)*ones(n,1);
 TT=norm(T,inf);
 if TT>=cutoff
     wild2=1;
 end
 if TT<cutoff
     wild2=0;
 end
end


