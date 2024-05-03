function y_est = cv_lambda_fn(thetas, y, ncv, lambda)
% return the estimate Y with ncv fold CV for tuning lambda 
% input 
%   thetas: pn x n x sn 
%   y: n x 1
%   ncv: number of CVs
%   lambda: penalty term 
% return
%   y_est: estimated Y: n x 1

    n = size(thetas, 2);
    cv_idxs = gen_cv_idxs(n, ncv);
    y_est = zeros(n, 1);
    for cvi = 1:ncv
        y_train = y(cv_idxs~=cvi);
        thetas_test = thetas(:, cv_idxs==cvi, :);
        thetas_train = thetas(:, cv_idxs~=cvi, :);
        [eta_est_cv, ~] = eta_est_wrapper(thetas_train, y_train, lambda);
        y_est_cv = recon_y(thetas_test, eta_est_cv);
        y_est(cv_idxs==cvi) = y_est_cv;
    end

end
