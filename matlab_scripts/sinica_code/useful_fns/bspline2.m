function [f,coeff,orthcoeff] = bspline2(k,n,knots,x,y,An)
% This functions gives the outputs as :
%Output1.f=B form of the fitted curve, 
%Output2.coeff=the coefficients for Basis ,    % this is a n by 1 vector
%Output3.orthcoeff=the coefficients for orthonormal B-spline basis(Basis1)    
f=spap2(knots,k,x,y);  % B form
coeff=f.coefs;
orthcoeff=coeff*An;
end











