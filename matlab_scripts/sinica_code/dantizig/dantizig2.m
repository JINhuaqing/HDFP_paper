function [lamdaopt, min_v, CV1]=dantizig2(Hn,theta,lamdaseq,GK1)
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
%% input variables
%input 1: Hn is used for specifying the null hypothesis, in row vectors, [1,3] means H0:beta1=beta3=0
%input 2: theta is the cell(1,pn) quantity, where pn is the number of
%         functional predictors, and each theta{j} is the n by sn centalized eigenscore matrix for
%         the j'th functional predictor, sn is the truncation size
%input 3: n is the sample size.
%input 4: sn is truncation size, often the optimal truncation size chosen
%         from the penalized regression procedure.
%input 5: lamdaseq is a row vector specifying the lamda values for tuning
%         during the dantizig procedure. (\tau_n in the paper). For e.g.,
%         lamdaseq=[0.01:0.01:0.1]
%input 6: GK1 is the setting of cross-validation for tuning \tau_n or
%         lamda, for e.g, we can set GK1=gen_cv_idxs(n, 5); where
%         5 means the five-folds-cv
%% output variables
%output 1: M is the hn*sn by (pn-hn)sn matrix=(\hat{w})'=transpose of \hat{w},  in the paper
%output 2: lamdaopt is the optimal lamda chosen from lamdaseq
%output 3: min_v is the minimum value of the CV loss function
%%
folds1=max(unique(GK1));% this is the number of folds for cv
m1=size(lamdaseq,2); %m1 is the column dim of lamdaseq which is the number of lamdas
pn=size(theta,2);
[n, sn] = size(theta{1});
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
        sol=dantizig1(Hn,thetacv,lamdaseq(r)); %this is the estimator from training sample,sol is hn*sn by (pn-hn)*sn matrix
        CV2=zeros(1,hn*sn);%%%for the loss for l=1,..,hnsn
        for rr=1:hn*sn
        CV2(rr)=(norm(ncv1^(-1)*(Theta22)'*(Theta22*sol(rr,:)'-Theta11(:,rr)),2))^2; %L2 norm loss
        end
        CV1(k,r)=sum(CV2);
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   end of cross validation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   end of cross validation
[min_v, min_loc]= min(mean(CV1));
lamdaopt=lamdaseq(min_loc); %optimal lamda'
%M=dantizig1(Hn,theta,lamdaopt);

    
        
   
    


end

