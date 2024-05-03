function y = subcell( x, index)
%% output
%output1:y is an cell(1,m) quantity such that y{j}=x{index(j)}
%% begin program
m=length(index);
y=cell(1,m);
for j=1:m
    y{j}=x{index(j)};
end
%% end program
end

