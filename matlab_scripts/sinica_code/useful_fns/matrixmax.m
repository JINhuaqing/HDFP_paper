function [I_row, I_col]= matrixmax(A)
%UNTITLED Summary of this function goes here
%  Input: matrix A
% output;[row,column] index of the largest element in A
[M,I] = max(A(:));

[I_row, I_col] = ind2sub(size(A),I);

end

