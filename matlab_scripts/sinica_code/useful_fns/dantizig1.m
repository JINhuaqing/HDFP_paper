function dantizig1=dantizig1(Hn,theta,n,sn,lamda)
%This is the dantizig procedure for column wisely optimzing
% w_l isn (pn-hn)sn by 1;  E_l is n by 1 vector, l=1,..,hn*sn

% \hat{w}_l=argmin||w_l||1, st, 
%       ||n^(-1)\theta_{Hn^c}'(\theta_{Hn^c}*w_l-E_l)||_{\infty}<=lamda
%   Let A=n^(-1/2)\theta_{Hn^c}, we want to get
% \hat{w}_l=argmin||w_l||1, st, 
%       ||A'(A*w_l-n^(-1/2)E_l)||_{\infty}<=lamda1, for each l=1,...,hnsn

%%we have \theta_{Hn}=[E_1,.., E_{hnsn}] is n by hn*sn matrix 
pn=size(theta,2);
Hnc=sort(setdiff(1:pn,Hn));
[m1,pn]=size(theta);
Theta=cell2mat(theta);%
[m2,hn]=size(Hn); % hn 
G=sort(repmat(1:pn,1,sn));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
%%Next we begin dantizig fitting
A=n^(-1/2)*Theta22;
M=zeros(hn*sn,(pn-hn)*sn);
for r=1:hn*sn
    y=n^(-1/2)*Theta11(:,r);
    [xk_1, lambdak_1, gamma_xk, gamma_lambdak, iter, th] = DS_homotopy_function(A, y, lamda);
    M(r,:)=(xk_1)';
end

dantizig1=M;


end

