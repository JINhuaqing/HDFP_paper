function lamdaseq= lamdaseq(a,b,N) 
kk=-5:(log(b-a)+5)/(N-1):log(b-a);
lamdaseq=a*ones(1,N)+exp(kk);
end
