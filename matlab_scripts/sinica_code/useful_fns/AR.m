function AR= AR( rho,n )
%UNTITLED3 Summary of this function goes here this
%generates the n by n AR correlation matrix, with rho \in (0,1), the
%(i,j)'th element is AR(i,j)=rho^{|i-j|}
%   Detailed explanation goes here
M=ones(n,n);
for i=1:n
    for j=1:n
        M(i,j)=rho^(abs(i-j));
    end
end
AR=M;

end

