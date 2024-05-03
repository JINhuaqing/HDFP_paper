function [knots,Basis,orthBasis,An] = bspline1(k,n,a,b)
% This functions gives the outputs as :1.knots sequence, 2.the B-spline basis , and the
%      3.orthonormal B-spline basis
knots=augknt([a:(b-a)/(n-k+1):b], k); 
Basis=cell(1,n);
for j=1:n
    Basis{j}=bspline(knots(j:j+k)); %%pp form
    Basis{j}=fn2fm(Basis{j},'B-');  %%B form
end
[orthBasis,An]=gram1(Basis,a,b);
end

% toy example:
% a=0;b=1;n=10;k=4;
% for j=1:n
% fnplt(Basis{j})
% hold on
% end


% M=zeros(n,n);
% for j=1:n
%     for k=1:n
%    product=fncmb(g{j},'*', g{k});  %pp form
%   product=fn2fm(product,'B-');  %%B form
%    a1=min(product.knots); %lower bound
%    b1=max(product.knots); % upper bound
%    f1=fnint(product);    %int_{a1}^{x}
%    integral=fnval(f1,b1);
%    M(j,k)=integral;
%     end
% end
   
   
   
%%%%caculate the integral of product
% for j=1:n
%     tt=Basis{j}; % B form 
%    product=fncmb(tt,'*', tt);  %pp form
%   product=fn2fm(product,'B-');  %%B form
%    a1=min(product.knots); %lower bound
%    b1=max(product.knots); % upper bound
%    f1=fnint(product);    %int_{a1}^{x}
%    integral=fnval(f1,b1)
% end