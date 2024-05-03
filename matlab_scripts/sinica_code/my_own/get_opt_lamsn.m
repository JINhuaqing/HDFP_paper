% GET_OPT_LAMSN returns the optimal values of SN and lambda index based on
% the input arrays of SN and lambda index values.
%
% [GOPT_SN, GOPT_LAM_IDX, ALL_MAX_KEYS] = GET_OPT_LAMSN(OPT_SNS, OPT_LAM_IDXS)
% returns the optimal values of SN and lambda index based on the input arrays
% of SN and lambda index values. It also returns all the maximum keys that
% have the same maximum count.
%
% Inputs:
%   - OPT_SNS: an array of SN values
%   - OPT_LAM_IDXS: an array of lambda index values
%
% Outputs:
%   - GOPT_SN: the optimal SN value
%   - GOPT_LAM_IDX: the optimal lambda index value
%   - ALL_MAX_KEYS: all the maximum keys that have the same maximum count
%
% Example:
%   opt_sns = [1, 2, 3, 1, 2, 3, 1, 2, 3];
%   opt_lam_idxs = [1, 1, 1, 2, 2, 2, 3, 3, 3];
%   [gopt_sn, gopt_lam_idx, all_max_keys] = get_opt_lamsn(opt_sns, opt_lam_idxs);
function [gopt_sn, gopt_lam_idx, all_max_keys] = get_opt_lamsn(opt_sns, opt_lam_idxs)
    % 
    countMap = containers.Map();
    for i = 1:length(opt_sns)
        combo = [num2str(opt_sns(i)) ',' num2str(opt_lam_idxs(i))];
        if isKey(countMap,combo)
            countMap(combo) = countMap(combo) + 1;
        else
            countMap(combo) = 1;
        end
    end
    
    allCounts = cell2mat(values(countMap));
    allKeys = keys(countMap);
    maxct = max(allCounts);
    all_max_idxs = find(allCounts==maxct);
    all_max_keys = {};
    flag = 1;
    for max_idx = all_max_idxs
        cur_max_key = allKeys{max_idx};
        cur_max_key = str2num(cur_max_key);
        all_max_keys{flag} = cur_max_key';
        flag = flag + 1;
    end
    all_max_keys = cell2mat(all_max_keys)';
    all_max_keys = sortrows(all_max_keys, [1, 2], {'ascend', 'descend'});
    gopt_sn =  all_max_keys(1,1);
    gopt_lam_idx = all_max_keys(1,2);
    
end