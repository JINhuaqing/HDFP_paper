function block2=block2(A,n)
%UNTITLED Summary of this function goes here
%   A is s by s matrix,  n>=2
% output =diag{A,...,A}, with n matrixes A
B=A;
for j=1:n-1
    B=blkdiag(B,A);
end
block2=B;
end

