function y_est = recon_y(thetas, eta_est)
% input
%   thetas: nroi x nsub x sn
%   eta_est: nroi x sn
% return 
%   y_est: estimated Y

    n = size(thetas, 2);
    y_est = zeros(n, 1);
    for sub_ix = 1:n
        yi_est = sum(squeeze(thetas(:, sub_ix,:)) .* eta_est, "all");
        y_est(sub_ix, 1) = yi_est;
    end


end