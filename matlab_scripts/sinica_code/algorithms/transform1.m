function eta11= transform1(sn,pn,theta3,f11)
%This is the transform function that transform the f11 from
%algolass1.m;algoscad1.m ;algomcp.m to the eta11 which is the estimates of
%coefficients


eta11=zeros(pn*sn,1);
G=sort(repmat(1:pn,1,sn));
for j=1:pn
eta11(G==j)=theta3{j}*f11{j};   % step vi
end
end

