function wild1= wild1(alpha,n,N,S)
%UNTITLED5 Summary of this function goes here This returns the cutoff value
%for testing
   e = mvnrnd(zeros(n,1),eye(n),N);%this is N by n marix, mean(e) will close to 0
   SS=cell2mat(S)*(e'); %%SS is the d by N, whose k'th column = Te=\sum_{j=1}^n ej*S{j}, for k=1,..,N
   SSS=ones(1,N);
   for j=1:N
       SSS(j)=norm(SS(:,j),inf);
   end
   wild1=quantile(SSS,1-alpha); %%%%%%%quantile cutoff ,if ||T||<wild1, we accept the Ho

end

