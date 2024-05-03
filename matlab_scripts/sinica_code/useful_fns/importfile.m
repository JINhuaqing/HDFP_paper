function mydata=importfile(index, numfiles)
%Refer to https://www.mathworks.com/help/matlab/import_export/process-a-sequence-of-files.html
%% This function aims to import a seq of files from a folder in increasing
%order of the indexs of files

%input1: index is the vector of labels of the files in folder in increasing order ; for example
%        10004,10026,10500,....
%input2: number of files in the folder
%% Note that the path to all files of the folder is prespecified before using this function
%for example 
%cd('C:/Users/admin/Desktop/matlab2014b/multpletestingYYF/realdata/HCPdata1/fMRI360parcellationgambling');
%% output: importfile is a cell(1, numfiles) quantity, where A{i} correspond to transpose of i'th file in the
%         folder

mydata=cell(1, numfiles);
for k = 1:numfiles
  myfilename = sprintf('subject_%d.dat', index(k));
  mydata{k} = load(myfilename);
  mydata{k}=mydata{k}'; %transpose for simplicity 253 by 360
end
%importfile=mydata;


end



%for example 
%cd('C:/Users/admin/Desktop/matlab2014b/multpletestingYYF/realdata/HCPdata1/fMRI360parcellationgambling');
%importfile=importfile(index, numfiles)
