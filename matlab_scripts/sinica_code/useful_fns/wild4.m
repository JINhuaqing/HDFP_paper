function [wild4, pvalue, TT, cutoff]= wild4(alpha,n,N,S)
%UNTITLED5 Summary of this function goes here This returns the pvalue 
%for testing 


%% begin program
   e = mvnrnd(zeros(n,1),eye(n),N);%this is N by n marix, mean(e) will close to 0
   SS=cell2mat(S)*(e'); %%SS is the d by N, whose k'th column = Te=\sum_{j=1}^n ej*S{j}, for k=1,..,N
   SSS=ones(1,N);
   for j=1:N
       SSS(j)=norm(SS(:,j),inf);
   end
   
   cutoff=quantile(SSS,1-alpha); %%%%%%%quantile cutoff=output4 ,if ||T||<wild1, we accept the Ho
   T=cell2mat(S)*ones(n,1);
 TT=norm(T,inf);
 if TT>=cutoff
     wild4=1;
 end
 if TT<cutoff
     wild4=0;
 end
   pvalue=pvalempir(SSS,TT);
end

