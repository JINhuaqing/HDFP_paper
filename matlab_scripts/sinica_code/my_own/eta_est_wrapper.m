function [eta_est, f_est] = eta_est_wrapper(thetas, y, lambda)
% input
%       thetas: an array pn x n x sn 
%       y: n x 1 vector, data
%      lambda: penalty term, number 
% return:
%      eta_est: estimate of eta sn x pn
%      f_est: not sure about this 

pn = size(thetas, 1); % num of ROIs
sn = size(thetas, 3); % num of basis
n = size(thetas, 2); % sample size 
G=sort(repmat(1:pn,1,sn));


theta1 = cell(1, pn);
for i = 1:pn
    theta1{1, i} = squeeze(thetas(i, :, :));
end

theta2 = cell(1, pn);
for i = 1:pn
    theta2{1, i} = theta1{i}*(inv(theta1{i}'*theta1{i}))*theta1{i}';
end

theta3 = cell(1, pn);
for i = 1:pn
    theta3{1, i} = (inv(theta1{i}'*theta1{i}))*theta1{i}';
end

eta = zeros(pn*sn,1);
f = cell(1, pn);
for i = 1:pn
    f{1, i} = zeros(n, 1);
end

[eta_est, f_est] = algoscad(y, n, sn, pn, lambda, G, theta1, theta2, theta3, eta, f);
eta_est = reshape(eta_est, [sn, pn])';

end