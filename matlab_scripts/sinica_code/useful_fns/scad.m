function scad=scad(lamda,a,t)
%%penalty function for scad in (t)=\rho_{lamda}(t|a)we usually set a=3.7, 
%%t \in R
if abs(t)<=lamda
    scad=lamda*abs(t);
end
if abs(t)>lamda&abs(t)<=a*lamda
    scad=(a*lamda*abs(t)-0.5*(t*t+lamda*lamda))/(a-1);
end
if abs(t)>a*lamda
    scad=lamda*lamda*(a+1)/2;
end
end

