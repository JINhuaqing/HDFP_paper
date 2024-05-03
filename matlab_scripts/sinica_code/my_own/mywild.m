function [pval, CV, TT]= mywild(alpha,N,S)
    %hypythesis testing, 
    % inputs:
    % alpha: significance level, 0.05, typically
    % N: number of simulations for bootstrap
    % S:
    % output:
    % pval: p-value reject Ho if pval < alpha
    % CV: critical value
    % TT: test statistic, reject H0 if TT > CV
    n = size(S, 2);
    e = mvnrnd(zeros(n,1),eye(n),N);%this is N by n marix, mean(e) will close to 0
    SS=cell2mat(S)*(e'); %%SS is the d by N, whose k'th column = Te=\sum_{j=1}^n ej*S{j}, for k=1,..,N
    SSS=ones(1,N);
    for j=1:N
        SSS(j)=norm(SS(:,j),inf);
    end

    % test stat
    T=cell2mat(S)*ones(n,1);

    TT=norm(T,inf); % test stat
    CV=quantile(SSS,1-alpha); %critival value ,if ||T||<CV, we accept the Ho
    pval = mean(SSS>TT); % p-value
    
    end
    
    