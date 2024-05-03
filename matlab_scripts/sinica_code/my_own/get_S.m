function S = get_S(Hn, eta_est, M, theta1, y)
% get_S: get the S set, which is prepared for hypothesis testing
% inputs:
%   Hn: The H0 set
%   eta_est: The estimated eta, a matrix of size (pn, sn)
%   M: estimate of w, a matrix of size (sn, xxx)
%   theta1: the X data, a cell of size (1, pn), each cell contains a matrix of size (n, sn)
%   y: the y data, a array of size (n, 1)
% outputs:
%   S: a cell of size (1, n)

    [pn, ~] = size(eta_est);
    n = size(y, 1);
    HnC = setdiff(1:pn, Hn);
    
    % get E cell
    E = cell(1, n); 
    Hn_theta1 = theta1(1, Hn);
    Hn_theta1_mat = cat(2, Hn_theta1{:});
    for i  = 1:n
        E{1, i} = Hn_theta1_mat(i, :)';
    end
    
    F = cell(1, n); 
    HnC_theta1 = theta1(1, HnC);
    HnC_theta1_mat = cat(2, HnC_theta1{:});
    for i  = 1:n
        F{1, i} = HnC_theta1_mat(i, :)';
    end
    
    Lambda = cell(1, pn);
    for i = 1:pn
        mat = theta1{1, i}.^2;
        Lambda{1, i} = mean(mat, 1);
    end
    Lambda_Hn = Lambda(1, Hn);
    Lambda_Hn_vec = cell2mat(Lambda_Hn);
    
    % select HnC from eta_est
    eta_est_HnC = cell(1, size(HnC, 2));
    for i = 1:length(HnC)
        eta_est_HnC{1, i} = eta_est(HnC(i), :);
    end
    eta_est_HnC_vec = cell2mat(eta_est_HnC);
    
    % Get  S
    S = cell(1, n);
    for i = 1:n
        right_part = y(i) - eta_est_HnC_vec * F{1, i};
        left_part = diag(1./Lambda_Hn_vec) * (M*F{1, i} - E{1, i});
        S{1, i} = left_part * right_part * n.^(-1/2);
    
     end

end