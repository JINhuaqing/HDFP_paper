function parsave(fname, result)
    % a counterpart of save fn for parfor
    % result is a struct
    save(fname, '-struct', 'result')
end