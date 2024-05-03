% INPUTS:
% cellArray - a cell array of matrices, 1 x cellSize

% OUTPUTS:
% ridx - the index of the row with the smallest mean value
% cidx_1se - the index of the column that is closest to the upper bound of the mean value plus one standard error
function [ridx, cidx_1se] = get_optlamsnidx_1se(cellArray)

cellSize = length(cellArray);
matrixSize = size(cellArray{1});

% Calculate the mean of each matrix in the cell array
meanArray = zeros(matrixSize);
for i = 1:cellSize
    meanArray = meanArray + cellArray{i};
end
meanArray = meanArray / cellSize;

% Calculate the standard deviation of each matrix in the cell array
stdArray = zeros(matrixSize);
for i = 1:cellSize
    stdArray = stdArray + (cellArray{i} - meanArray).^2;
end
stdArray = sqrt(stdArray / (cellSize - 1));
seArray = stdArray / sqrt(cellSize);

% Find the row with the smallest mean value
[ridx, cidx] = matrixmin(meanArray);

% Find the upper bound of the mean value plus one standard error
sel_row = meanArray(ridx, :);
upbd = meanArray(ridx, cidx) + seArray(ridx, cidx);

% Find the index of the column that is closest to the upper bound
cidx_1se = max(find(sel_row <= upbd));
end
