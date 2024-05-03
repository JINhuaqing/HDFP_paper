function meancell = meancell(a)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
% input: a is an cell(1,n), with a{j} as a m by p matrix,   n>=1
% output: meancell=mean of elements in a 
n=size(a,2);
if n>1
M=a{1};
for j=1:n-1
    M=M+a{j+1};  % sum operation
end
meancell=M/n; % mean operation
end

if n==1
    meancell=a{1};
end


end

