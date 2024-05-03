function [eta11, f11]=algoscad(y,n,sn,pn,lamda,G,theta1,theta2,theta3,eta,f)
%This is the algorithm for scad penalty for penalized regression
%% input variables
%input 1: y is the n by 1 response vector, centalized in advance.
%input 2: n is the sample size
%input 3: sn is the truncation size
%input 4: pn is the number of functinal predictors
%input 5: lamda is the penalty parameter \lambda_n in the paper
%input 6:  G=sort(repmat(1:pn,1,sn));
%input 7: theta1 is the cell(1,pn) quantity, where pn is the number of
%         functional predictors, and each theta{j} is the n by sn centalized eigenscore matrix for
%         the j'th functional predictor, sn is the truncation size
%input 8: theta2 is the cell(1,pn) quantity, such that theta2{j}=theta1{j}*(inv(theta1{j}'*theta1{j}))*theta1{j}';
%input 9: theta3 is the cell(1,pn) quantity, such that theta3{j}=(inv(theta1{j}'*theta1{j}))*theta1{j}'
%input 10: eta is an pnsn by 1 vector, serving as the initial estimate of
%          (\eta_1',...,\eta_{p_n}')' on page 10 of the paper, we can initially set
%          eta=zeros(pn*sn,1) and then set the eta as the one calculated from the proximal lamda
%          value under the same sn
%input 11: f is a cell(1,pn) such that f{j} is a n by 1 vector,  we can initially set
%          f{j}=zeros(n,1) and then set the f{j} as the one calculated from the proximal lamda
%          value under the same sn, (i.e., f11). In fact, each
%          f{j}=\hat{f}_j, where \hat{f}_j is defined on page 31 of the
%          paper
%% output variables
%output 1: eta11 is an pn*sn by 1 vector, which is the final penalized estimator for \eta given sn and lamda
%output 2: f11 is the final estimator for f given sn and lamda
%% note that the optimal sn and lamda can be chosen by cross-validation using square loss
distance=1;
H=1:pn;
f1=f;
R1=cell(1,pn);
P1=cell(1,pn);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%get f11
while distance>=0.0001
for j=1:pn
    R1{j}=y-[f{H~=j}]*ones(pn-1,1);% step ii
    P1{j}=theta2{j}*R1{j};         % step iii
    %f{j}=max(0,1-scadderi(lamda*sqrt(sn),3.7,n^(-1/2)*norm(f{j}))*sqrt(n)/norm(P1{j}))*P1{j}; % step iv
    f{j}=max(0,1-scadderi(lamda*sqrt(sn),3.7,n^(-1/2)*norm(f{j}))*sqrt(n)/(norm(P1{j})+1-logical(norm(P1{j}))))*P1{j}; % step iv
    f{j}=f{j}-n^(-1)*ones(1,n)*f{j}*ones(n,1); % step v 
end
distance=trace((cell2mat(f1)-cell2mat(f))'*(cell2mat(f1)-cell2mat(f)))/n;
f1=f;
end
f11=f1;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




%%%%%%%%%%%%%%%%%%%%%% get eta11
for j=1:pn
eta(G==j)=theta3{j}*f1{j};   % step vi
end
eta11=eta;
%%%%%%%%%%%%%%%%%%%%%%%
%sort(repmat(1:pn,1,sn))% this fuction gives the pn*sn dimensional vector G=[ones(1,sn),2*ones(1,sn),..,pn*ones(1,sn)]
end






