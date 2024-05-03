function cv_indices = gen_cv_idxs(n, ncv)
    % 创建一个空的矩阵来存储 cv_indices
    cv_indices = zeros(n, 1);
    
    % 计算每一折（fold）应该有的元素数量
    base_num = floor(n / ncv);
    
    % 计算剩余的元素数量
    residual = n - base_num * ncv;
    
    % 为每一折分配元素
    for i = 1:ncv
        if i <= residual
            cv_indices((i-1) * base_num + i : i * base_num + i) = i;
        else
            cv_indices((i-1) * base_num + residual + 1 : i * base_num + residual) = i;
        end
    end
    
    % 随机排序
    cv_indices = cv_indices(randperm(n));
end