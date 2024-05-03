function f11=algoscad1(y,n,sn,pn,lamda,theta2,f)
%This is the algorithm for scad penalty but only return f11


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
    f{j}=max(0,1-scadderi(lamda*sqrt(sn),3.7,n^(-1/2)*norm(f{j}))*sqrt(n)/norm(P1{j}))*P1{j}; % step iv
    f{j}=f{j}-n^(-1)*ones(1,n)*f{j}*ones(n,1); % step v 
end
distance=trace((cell2mat(f1)-cell2mat(f))'*(cell2mat(f1)-cell2mat(f)))/n;
f1=f;
end
f11=f1;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%





%sort(repmat(1:pn,1,sn))% this fuction gives the pn*sn dimensional vector G=[ones(1,sn),2*ones(1,sn),..,pn*ones(1,sn)]
end

