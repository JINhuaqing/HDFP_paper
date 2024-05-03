clear all;close all;clc;
data_name = "DK_timecourse";
file_name = "../data/AD_vs_Ctrl_ts/" + data_name + ".mat";
data = load(file_name);



fmin = 2; %Hz
fmax = 35;
fs=600;
nfreqs = 100;

all = struct;
for iy = 1:1
    
    key = "data_" + iy;
    cur_dat = data.DK_timecourse;
    %cur_dat = squeeze(data.dk10(iy, :, :));
    key
    pow_data = struct;
    pow_data.mat = zeros(68, nfreqs);
for ix = 1:68
    
[q, fsamples] = get_spectral(squeeze(cur_dat(ix, :)), fs, nfreqs, fmin, fmax);
pow_data.mat(ix, :) = q;

end
pow_data.freq = fsamples;
all.(key) = pow_data;
end

save_path = "../data/AD_vs_Ctrl_ts/" + data_name + nfreqs + "_PSD.mat";
save(save_path, "-struct", "all");
