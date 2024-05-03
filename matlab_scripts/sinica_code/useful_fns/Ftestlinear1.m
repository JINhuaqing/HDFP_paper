function pvalconclu = Ftestlinear1(y,M,submodels,Fullmodel,alpha)

%% begin program
vn=size(submodels,2);
pvalconclu=zeros(5,vn);
for j=1:vn
    [pvalue, conclusion, df1, df2, Fstatistic] = Ftestlinear(y,M,submodels{j},Fullmodel,alpha);
    pvalconclu(1,j)=pvalue;
    pvalconclu(2,j)=conclusion;
    pvalconclu(3,j)=df1;
    pvalconclu(4,j)=df2;
    pvalconclu(5,j)=Fstatistic;
end
%% end program

end


