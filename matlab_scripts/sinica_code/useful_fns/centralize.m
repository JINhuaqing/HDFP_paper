function sX = centralize(X)
[n,p] = size(X);
sX = X-repmat(mean(X),n,1);
end

