function block1=block1(A,n)
%UNTITLED Summary of this function goes here
%   A is s by s matrix,  n>=2
% output =diag{A,...,A}, with n matrixes A
B=A;
for j=1:floor(log(n)/log(2))
    B=blkdiag(B,B);
end
if 2^(floor(log(n)/log(2)))==n
block1=B;
end
if n-2^(floor(log(n)/log(2)))==1
block1=blkdiag(B,A);
end

if n-2^(floor(log(n)/log(2)))>1
mm=block2(A,n-2^(floor(log(n)/log(2))));
block1=blkdiag(B,mm);
end

end


