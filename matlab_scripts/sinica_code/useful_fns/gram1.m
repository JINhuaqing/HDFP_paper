function [g,An] = gram1(f,a,b)


[nn,n]=size(f);
An=zeros(n,n);
g=cell(1,n);
for j=1:n
    if j==1
            tt=f{j}; % B form 
   product=fncmb(tt,'*', tt);  %pp form
  product=fn2fm(product,'B-');  %%B form
   a1=min(product.knots); %lower bound
   b1=max(product.knots); % upper bound
   f1=fnint(product);    %int_{a1}^{x}
   integral=sqrt(fnval(f1,b1));
   An(j,j)=integral;
   f11=fncmb(tt,'*', 1/integral);%pp form
   f11=fn2fm(f11,'B-');  %%B form
   g{j}=f11;
    end
    
    if j>1
         tt=f{j}; % B form
        for k=1:j-1
   product=fncmb(f{j},'*', g{k});  %pp form
  product=fn2fm(product,'B-');  %%B form
   a1=min(product.knots); %lower bound
   b1=max(product.knots); % upper bound
   f1=fnint(product);    %int_{a1}^{x}
   integral=fnval(f1,b1);
   An(j,k)=integral;
   ss=fncmb(g{k},'*', integral);  %pp form
   ss=fn2fm(ss,'B-');  %%B form
   tt=fncmb(tt,'-', ss);  %pp form
   tt=fn2fm(tt,'B-');  %%B form
        end
        product=fncmb(tt,'*', tt);  %pp form
  product=fn2fm(product,'B-');  %%B form
   a1=min(product.knots); %lower bound
   b1=max(product.knots); % upper bound
   f1=fnint(product);    %int_{a1}^{x}
   integral=sqrt(fnval(f1,b1));
        An(j,j)=integral;   %%%*****
   f11=fncmb(tt,'*', 1/integral);%pp form
   f11=fn2fm(f11,'B-');  %%B form
   g{j}=f11;   
    end
end
            

   

end

