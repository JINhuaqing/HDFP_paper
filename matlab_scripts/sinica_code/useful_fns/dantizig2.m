function [M,lamdaopt]=dantizig2(Hn,theta,n,sn,lamdaseq,GK1)
%This is the dantizig procedure for column wisely optimzing
% w_l isn (pn-hn)sn by 1;  E_l is n by 1 vector, l=1,..,hn*sn

% \hat{w}_l=argmin||w_l||1, st, 
%       ||n^(-1)\theta_{Hn^c}'(\theta_{Hn^c}*w_l-E_l)||_{\infty}<=lamda
%   Let A=n^(-1/2)\theta_{Hn^c}, we want to get
% \hat{w}_l=argmin||w_l||1, st, 
%       ||A'(A*w_l-n^(-1/2)E_l)||_{\infty}<=lamda1, for each l=1,...,hnsn

%%we have \theta_{Hn}=[E_1,.., E_{hnsn}] is n by hn*sn matrix 

%while the the optimal lamda is choosen from a sequence of lamda values
%lamdaseq


folds1=max(unique(GK1));% this is the number of folds for cv
m1=size(lamdaseq,2); %m1 is the column dim of lamdaseq which is the number of lamdas
pn=size(theta,2);
hn=size(Hn,2); % hn 
Hnc=sort(setdiff(1:pn,Hn));
G=sort(repmat(1:pn,1,sn));
%%%%%%%%%%%%%%%%%%%%%%%%%  we start CV as follows
CV1=zeros(folds1,m1); %this is the evaluation matrix for cross validations
for k=1:folds1
    %%%%%%prepare the inputs for k'th fold, k'th fold is used as the
    %%%%%%testing set
    %Hn,sn is already known and is kept the same
    %for the theta, the cell(1,pn) quantity, we replace it by thetacv(the
    %training set
    thetacv=cell(1,pn); %this is training
    thetacv1=cell(1,pn);%this is testing
    for j=1:pn
        thetacv{j}=theta{j}(k~=GK1,:);
        thetacv1{j}=theta{j}(k==GK1,:);
    end
    %for sample size, we replace n by ncv
    ncv=size(thetacv{1},1); % size for training
    ncv1=n-ncv;             %size for testing
    %%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% fot testing set
    Theta=cell2mat(thetacv1);
Theta1=cell(1,hn); %This is Theta_{Hn}
for j=1:hn
    Theta1{j}=Theta(:,G==Hn(j));
end
Theta11=cell2mat(Theta1); % \theta_{Hn}=[E_1,.., E_{hnsn}] is n by hn*sn matrix  in matrix form
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Theta2=cell(1,pn-hn); %This is Theta_{Hn^c}
for j=1:(pn-hn)
    Theta2{j}=Theta(:,G==Hnc(j));
end
Theta22=cell2mat(Theta2); 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    for r=1:m1  %this is for the lamda sequence lamdaseq(r)
        sol=dantizig1(Hn,thetacv,ncv,sn,lamdaseq(r)); %this is the estimator from training sample,sol is hn*sn by (pn-hn)*sn matrix
        CV2=zeros(1,hn*sn);%%%for the loss for l=1,..,hnsn
        for rr=1:hn*sn
        CV2(rr)=(norm(ncv1^(-1)*(Theta22)'*(Theta22*sol(rr,:)'-Theta11(:,rr)),2))^2; %L2 norm loss
        end
        CV1(k,r)=sum(CV2);
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   end of cross validation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   end of cross validation
[I_row, I_col]= matrixmin(mean(CV1));
lamdaopt=lamdaseq(I_col); %optimal lamda'
M=dantizig1(Hn,theta,n,sn,lamdaopt);

    
        
   
    


end

