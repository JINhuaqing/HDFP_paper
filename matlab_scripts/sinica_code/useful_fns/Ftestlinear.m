function [pvalue, conclusion, df1, df2, Fstatistic] = Ftestlinear(y,M,model1,model2,alpha)

%% program begin
pn=size(M,2);
n=length(y);
%% case1: if model1 is nonempty
if isempty(model1)==0
    qn=length(model1);
    rn=length(model2);
    M1=cell(1,qn+1); %cell form of design matrix for model1
    M1{1}=ones(n,1);
    M2=cell(1,rn+1); %cell form of design matrix for model2
    M2{1}=ones(n,1);
    for j=1:qn 
        M1{j+1}=M{model1(j)};
    end
    for k=1:rn 
        M2{k+1}=M{model2(k)};
    end
    M11=cell2mat(M1); %matrix form of design matrix for model1
    M22=cell2mat(M2); %matrix form of design matrix for model2
    df1=n-size(M11,2);  %degree of freedom for model1=df1>df2
    df2=n-size(M22,2);  %degree of freedom for model2=df2
    beta11=regress(y,M11);
    beta22=regress(y,M22);
    SS1=(y-M11*beta11)'*(y-M11*beta11);       
    SS2=(y-M22*beta22)'*(y-M22*beta22);
    Fstatistic=(SS1-SS2)/(df1-df2)/(SS2/df2);  %%%%%%
    pvalue=1-fcdf(Fstatistic,df1-df2,df2);     %%%%%%
    if pvalue<alpha
        conclusion=1; % we choose model2
    end
    if pvalue>=alpha
          conclusion=0; % we choose model1
    end  
end
%% case2: if model1 is empty
if isempty(model1)==1
    
    rn=length(model2);
    M1=cell(1,1); %cell form of design matrix for model1
    M1{1}=ones(n,1);
    M2=cell(1,rn+1); %cell form of design matrix for model2
    M2{1}=ones(n,1);
    
    for k=1:rn 
        M2{k+1}=M{model2(k)};
    end
    M11=cell2mat(M1); %matrix form of design matrix for model1
    M22=cell2mat(M2); %matrix form of design matrix for model2
    df1=n-size(M11,2);  %degree of freedom for model1=df1>df2
    df2=n-size(M22,2);  %degree of freedom for model2=df2
    beta11=regress(y,M11);
    beta22=regress(y,M22);
    SS1=(y-M11*beta11)'*(y-M11*beta11);       
    SS2=(y-M22*beta22)'*(y-M22*beta22);
    Fstatistic=(SS1-SS2)/(df1-df2)/(SS2/df2);  %%%%%%
    pvalue=1-fcdf(Fstatistic,df1-df2,df2);     %%%%%%
    if pvalue<alpha
        conclusion=1; % we choose model2
    end
    if pvalue>=alpha
          conclusion=0; % we choose model1
    end  
end
%% end of program

end

