function [n,knots,Basis,orthBasis,An] = bspline3(k,numinknots,a,b)
% This functions gives the outputs as :1.knots sequence, 2.the B-spline basis , and the
%      3.orthonormal B-spline basis
n=numinknots+k;
knots=augknt([a:(b-a)/(n-k+1):b], k); 
Basis=cell(1,n);
for j=1:n
    Basis{j}=bspline(knots(j:j+k)); %%pp form
    Basis{j}=fn2fm(Basis{j},'B-');  %%B form
end
[orthBasis,An]=gram1(Basis,a,b);
end

