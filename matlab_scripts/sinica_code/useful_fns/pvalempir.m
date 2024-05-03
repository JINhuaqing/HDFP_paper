function pvalempir = pvalempir(S,T)
%% begin program
N=length(S);
M=length(find(S>=T));
pvalempir=M/N;



end

