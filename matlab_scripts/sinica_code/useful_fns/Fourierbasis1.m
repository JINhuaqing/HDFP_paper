function Fourierbasis1=Fourierbasis1(a,b,m,n)
%This generates Fourierbasis evaluation matrix on grids mgrids(a,b,m)=[t1,...,tm]
M=ones(n,m)*sqrt(1/(b-a));
mgridss=mgrids(a,b,m);
if mod(n,2)==0
    for k=1:(n/2)
        for j=1:m
            M(2*k,j)=sqrt(2/(b-a))*cos(k*pi*(2*mgridss(j)-a-b)/(b-a));
        end
    end
    for k=2:(n/2)
        for j=1:m
            M(2*k-1,j)=sqrt(2/(b-a))*sin((k-1)*pi*(2*mgridss(j)-a-b)/(b-a));
        end
    end
end



if mod(n,2)>0
    for k=1:(n-1)/2
        for j=1:m
            M(2*k,j)=sqrt(2/(b-a))*cos(k*pi*(2*mgridss(j)-a-b)/(b-a));
        end
    end
    for k=2:(n+1)/2
        for j=1:m
            M(2*k-1,j)=sqrt(2/(b-a))*sin((k-1)*pi*(2*mgridss(j)-a-b)/(b-a));
        end
    end
end
        
Fourierbasis1=M;
end



