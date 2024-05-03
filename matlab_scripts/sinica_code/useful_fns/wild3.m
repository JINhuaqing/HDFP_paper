function [wild2, TT, cutoff]= wild3(alpha,n,N,S)
% This returns the output for
%testing
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



