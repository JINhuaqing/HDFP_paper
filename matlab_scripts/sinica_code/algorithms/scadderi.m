function scadderi=scadderi(lamda,a,t)
%%derivative of the penalty function for scad in (t)=\rho'_{lamda}(t|a)we usually set a=3.7, 
%%we assume t>=0, in the setting
if t<=lamda;
   scadderi=lamda;
end
if t>lamda && t<=a*lamda;
    scadderi=(a*lamda-t)/(a-1);
end
if t>a*lamda;
    scadderi=0;
end
end
