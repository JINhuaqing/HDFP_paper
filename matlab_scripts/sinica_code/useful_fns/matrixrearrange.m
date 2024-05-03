function matrixrearrange = matrixrearrange(A, index1,index2)
%UNTITLED3 此处显示有关此函数的摘要
%   此处显示详细说明
%input1: A is an n by m matrix
%input2; index1 is a 1 by n vector like index1=[1,1,1,2,2,3,3,3,3]
%input3: index2 is the unique index row vector we want for the output, such
%       as index2=[1,2] or index2=[1,3] or index2=[3,1]
%output1: This is the matrix B such that
%         the j th row of B is equal to A(index1==index2(j),:)
B=A(index1==index2(1),:);
n=length(index2);
for j=2:n
    B=[B;A(index1==index2(j),:)];
end
 matrixrearrange=B;  
    


end

