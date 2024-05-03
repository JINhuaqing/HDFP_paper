function [A, B] = regresscvsnj(y,cv,M,Mr)
%UNTITLED2 此处显示有关此函数的摘要
%   此处显示详细说明 fixed crossvalidation(ie, cross validation is known in
%   advance.)
% input1: y is an n by 1 response vector
% input2: cv is the pattern of crossvalidation, such as
%         [1,1,1,1,2,3,2,3,2,3] for a 3 folds cross validation for n=10 .
%         for example, we let cv=crossvalind('Kfold', n, 5);
%          ignore: { if cv=[ ], we randomly generate it inside the function based on
%                    the folds provided, and we always set in this way}
% input3: M is the 1 by pn(>=2) cell quantity, such that M{j} is the 
%         n by Smaxj(the max number of thruncation size=number of basis=30)
%         matrix of design matrix corresponding to j'th predictor
% input4: Mr is the 1 by pn cell quantity, such that Mr{j} is the 
%        1 by srj vector such as [2,3,..,Smaxj-2] specifying the possible
%        range of truncation sizes for j'th predictor for selection, usually,
%        Mr{j}=[1,..,Smaxj]
%%
% output1:A=[sn1, sn2,..,snj,..,snpn] is an 1 by pn vector of truncation sizes chosen by cross validation by
%         regression and use square loss to measure
% output2:B =is an sum(A) by 1 vector of  estimated regression coefficients
%         based on whole data
n=length(y);                                           %this is sample size
pn=size(M,2);                                          %this is the # of predicors
folds=length(unique(cv));                              % # of folds for cv

% Mr1=cell(1,pn);                                        %Mr1{j}=Smaxj in general
% for i=1:pn
%     Mr1{i}=length(Mr{i});
% end


Mr1=combvec(Mr{1},Mr{2});                               %Mr1 is the pn*pn1 matrix,each column is one case of snj combination
if pn>2
for j=1:pn-2
    Mr1=combvec(Mr1,Mr{j+2});
end
end

pn1=size(Mr1,2);
%pn1=prod(cell2mat(Mr1));
Mr2=cell(1,pn1);                                        %Mr2{k} is the kth design matrix for fitting
for k=1:pn1
    aux1=Mr1(:,k);
    aux2=cell(1,pn); %%**
    for ks=1:pn
        aux3=M{ks};
        aux2{ks}=aux3(:,1:aux1(ks));
    end
    Mr2{k}=centralize(cell2mat(aux2));
end

%% Next we begin fitting
%CV=ones(1,pn1);         %this is the evaluation score matrix for the crossvalidation
CV1=ones(folds,pn1);    % CV=mean(CV1)
y=centralize(y);
for ks1=1:pn1
    for ks2=1:folds
        ytest=y(cv==ks2);
        ytrain=y(cv~=ks2);
        ytest=centralize(ytest);
        ytrain=centralize(ytrain);
        %%%%%%%%%%%%%%
        Maux=Mr2{ks1};
        %Maux=centralize(Maux);
        Mtest=centralize(Maux(cv==ks2,:));
        Mtrain=centralize(Maux(cv~=ks2,:));
        %%%%%%%%%%%%%
        betatrain=regress(ytrain,Mtrain);
        score=(ytest-Mtest*betatrain)'*(ytest-Mtest*betatrain);
        CV1(ks2,ks1)=score;
        %regress(y,Mr2{ks1}
    end
end
CV=mean(CV1); %%%%%%this is the evaluation score matrix for the crossvalidation
A=(Mr1(:,find(CV==min(CV))))';
B=regress(y,Mr2{find(CV==min(CV))});




end

