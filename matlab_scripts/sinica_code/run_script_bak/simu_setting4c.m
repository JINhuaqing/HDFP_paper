% Run simulation on the code the authors provided
% use it as a template to get other script.
clear all;
% always use this working directory
%cd /data/rajlab1/user_data/jin/MyResearch/HDF_infer/matlab_scripts/
%cd '/Users/hujin/Library/CloudStorage/OneDrive-UCSF/Documents/ProjectCode/HDF/matlab_scripts'
cd '/wynton/home/rajlab/hjin/MyResearch/HDF/matlab_scripts/'

addpath sinica_code/my_own/
addpath sinica_code/dantizig/
addpath sinica_code/algorithms/
addpath sinica_code/wild_bootstrap/
addpath sinica_code/useful_fns/

%% 1.The setting is as follows
sn_upper = 30; % upper bd of sn

%% 1.1 path
root_data_folder = '../mid_results/matlab_simu_data/';
data_folder = 'SinicaX_d-200_n-100_setting4c/';

% setting when c1=0, H0, c1>0 H1
Hn = [1];
c1 = 0.40;
num_rep = 200; 

% m: equally spaced points on [a,b]
m = 100;

%% 1.2 Dantizig
folds_d=5; 
lams_d = lamdaseq(0, 16, 17); % lam seqs for dantizig,
%lams_d = lamdaseq(0, 0.4, 10); % lam seqs for dantizig,

%% 1.3 eta estimate
sns = 1:3:sn_upper; % candidate seq of sn
folds=5; 
upper=16;
n_lam =17; 
lams_eta =lamdaseq(0,upper,n_lam); % lam seq for eta

a=0; b=1; % the funcitional curve is defined on [a,b]=[0,1]
[knots,Basis,orthBasis,An] = bspline1(4, sn_upper, a, b); % require additional toolbox (on Aug 16, 2023)
grids=mgrids(a,b,m);



%% 2. Get opt lam and sn
CVs = cell(1, num_rep);
%opt_lam_idxs = zeros(1, num_rep);
%opt_sns = zeros(1, num_rep);
% parallel runing
parpool(25);
parfor rep_ix = 1:num_rep
    %% 1.4 data
    %rep_ix = 1;
    fil_name = ['c1_' num2str(c1*1000) '_seed_' num2str(rep_ix-1)];
    fil_name
    cur_data = load([root_data_folder data_folder fil_name '.mat']);
    [n, pn, ~] = size(cur_data.X_centered);
    
    
    
    %% 2.Next, we process the X and Y
    y = cur_data.Y_centered';
    
    %cv_idxs =crossvalind('Kfold', n, folds); % 'folds=5' folds for sample size n data, correspond to a single monte carlo for lamdan , sn
    cv_idxs =gen_cv_idxs(n, folds); % 'folds=5' folds for sample size n data, correspond to a single monte carlo for lamdan , sn
    
    % Next we generate the f, coeffs and orthcoeffes for each realizations xij(.) on m=100 equally spaced points on [a,b]
    % i..e., get the basis (on Aug 18, 2023), in fact, M1 and M2 are no use
    M1=cell(n,pn);
    M2=cell(n,pn);
    M3=cell(n,pn); % basis
    for i=1:n
        MM1= squeeze(cur_data.X_centered(i, :, :));
        for j=1:pn
            [f,coeff,orthcoeff]=bspline2(4, sn_upper, knots, grids, MM1(j,:), An);
            M1{i,j}=f;
            M2{i,j}=coeff;
            M3{i,j}=orthcoeff;
        end
    end
    
    %% Next we generate the design matrix M4=cell(1,pn), i.e., theta
    % The output M4=cell(1,pn), where
    % M4{j}=n by sn_upper(sn_upper is number of bspline basis=100 for example) matrix,
    % similar to \theta_j except sn is changed to sn_upper
    M4=cell(1,pn); % theta, in total, note that we may not need all sn_upper theta
    MM4=cell2mat(M3);
    GM4=sort(repmat(1:pn,1, sn_upper));
    for j=1:pn
        M4{j}=centralize(MM4(:,GM4==j));
    end
    
    %% Next we generate the estimated variance row vectors for the \theta_{ijk}
    M5=cell(1,pn);
    for j=1:pn
        M5{j}=(diag(M4{j}'*M4{j}/n))';
    end
    % end of generate the estimated variance row vectors for the \theta_{ijk}
    
    
    %% 3 Simulation
    %% 3.1 Get optimal sn and lam for eta
    CV1=zeros(length(sns),n_lam);
    opt_lam_idx =0;  %optimal lamda idx for eta estimator
    opt_sn =0;  %optimal number of bspline basis functions sn
    
    for sn_idx=1:length(sns)  %row of CV1
        sn = sns(sn_idx);
        G=sort(repmat(1:pn,1,sn));
        Theta1=cell(1,pn);
        for j=1:pn
            Theta1{j}=M4{j}(:,1:sn);
        end
        CV5=zeros(folds,n_lam);%%this is the crossvalid evaluation matrix,for 5 folds when we have sn basis function over m3 lamdas
        CV6=cell(folds,n_lam); %%This is the estimated eta for the training set
        for kf=1:folds  %ks means kfolds
            Theta11=cell(1,pn); %This is traing set for kf'th fold
            Theta22=cell(1,pn); %This is traing set for kf'th fold
            Theta33=cell(1,pn); %This is traing set for kf'th fold
            Theta111=cell(1,pn); %This is the testing set for kf'th fold
            for j=1:pn
                Theta11{j}=centralize(Theta1{j}(kf~=cv_idxs,:)); %training
                Theta22{j}=Theta11{j}*(inv(Theta11{j}'*Theta11{j}))*Theta11{j}'; %training
                Theta33{j}=(inv(Theta11{j}'*Theta11{j}))*Theta11{j}'; %training
                Theta111{j}=centralize(Theta1{j}(kf==cv_idxs,:));%testing
            end
            y11=centralize(y(kf~=cv_idxs,:)); %traing for response vector
            [n11,v11]=size(y11);         %n11 is the size for training
            y111=centralize(y(kf==cv_idxs,:)); %testing for response vector
            [n111,v111]=size(y111);      %n111 is the size for testing
            
            for ka=1:n_lam   %column of CV1  ka means index for lams_eta(ka) for ka=1,,..m3, this loop fix the kf'th row of CV5
                if ka==1
                    eta=zeros(pn*sn,1);   %%initial
                    f=cell(1,pn);
                    for kb=1:pn
                        f{kb}=zeros(n11,1);   %%initial
                    end
                    [eta11, f11]=algoscad(y11,n11,sn,pn,lams_eta(ka),G,Theta11,Theta22,Theta33,eta,f);
                    CV6{kf,ka}=eta11;
                    CV5(kf,ka)=(y111-cell2mat(Theta111)*eta11)'*(y111-cell2mat(Theta111)*eta11);
                    eta=eta11;
                    f=f11;
                end
                %%%%%%%%%%%%%%%%%%%%
                if ka>1
                    [eta11, f11]=algoscad(y11,n11,sn,pn,lams_eta(ka),G,Theta11,Theta22,Theta33,eta,f);
                    CV6{kf,ka}=eta11;
                    CV5(kf,ka)=(y111-cell2mat(Theta111)*eta11)'*(y111-cell2mat(Theta111)*eta11);
                    eta=eta11;
                    f=f11;
                end
                %ka
            end
        end
        CV1(sn_idx,:)=mean(CV5);
    end
    % end of cross validation
    [I_row, I_col]= matrixmin(CV1);
    %opt_lam_idx = I_col;  %optimal index of lamda
    %opt_sn=sns(I_row);  %optimal number of bspline basis functions sn
    %opt_sns(rep_ix) = opt_sn;
    %opt_lam_idxs(rep_ix) = opt_lam_idx;
    CVs{rep_ix} = CV1;
    
end
delete(gcp('nocreate'));






%% 3. Extract the optimal sn and lam
% [gopt_sn, gopt_lam_idx, ~ ] = get_opt_lamsn(opt_sns, opt_lam_idxs);
[gopt_sn_idx, gopt_lam_idx] = get_optlamsnidx_1se(CVs);
gopt_sn = sns(gopt_sn_idx);


%% 4. Get the opt lam_d under best sn and lam
opt_lams_d = zeros(1, num_rep);
% parallel runing
parpool(25);
parfor rep_ix = 1:num_rep
    %% 1.4 data
    %rep_ix = 1;
    fil_name = ['c1_' num2str(c1*1000) '_seed_' num2str(rep_ix-1)];
    fil_name
    cur_data = load([root_data_folder data_folder fil_name '.mat']);
    [n, pn, ~] = size(cur_data.X_centered);
    
    
    
    %% 2.Next, we process the X and Y
    y = cur_data.Y_centered';
    
    %cv_idxs_d =crossvalind('Kfold', n, folds_d); % 'folds=5' folds for sample size n data, correspond to a single monte carlo for lamda' of dantizig
    cv_idxs_d =gen_cv_idxs(n, folds_d); % 'folds=5' folds for sample size n data, correspond to a single monte carlo for lamda' of dantizig
    
    % Next we generate the f, coeffs and orthcoeffes for each realizations xij(.) on m=100 equally spaced points on [a,b]
    % i..e., get the basis (on Aug 18, 2023), in fact, M1 and M2 are no use
    M1=cell(n,pn);
    M2=cell(n,pn);
    M3=cell(n,pn); % basis
    for i=1:n
        MM1= squeeze(cur_data.X_centered(i, :, :));
        for j=1:pn
            [f,coeff,orthcoeff]=bspline2(4, sn_upper, knots, grids, MM1(j,:), An);
            M1{i,j}=f;
            M2{i,j}=coeff;
            M3{i,j}=orthcoeff;
        end
    end
    
    %% Next we generate the design matrix M4=cell(1,pn), i.e., theta
    % The output M4=cell(1,pn), where
    % M4{j}=n by sn_upper(sn_upper is number of bspline basis=100 for example) matrix,
    % similar to \theta_j except sn is changed to sn_upper
    M4=cell(1,pn); % theta, in total, note that we may not need all sn_upper theta
    MM4=cell2mat(M3);
    GM4=sort(repmat(1:pn,1, sn_upper));
    for j=1:pn
        M4{j}=centralize(MM4(:,GM4==j));
    end
    
    %% Next we generate the estimated variance row vectors for the \theta_{ijk}
    M5=cell(1,pn);
    for j=1:pn
        M5{j}=(diag(M4{j}'*M4{j}/n))';
    end
    % end of generate the estimated variance row vectors for the \theta_{ijk}
    
    
    %% Get eta estimator based on optimal sn and lam
    % The author use a loop from small to lager lams to get the good results
    G=sort(repmat(1:pn,1,gopt_sn));
    Theta1=cell(1,pn);
    Theta2=cell(1,pn);
    Theta3=cell(1,pn);
    for j=1:pn
        Theta1{j}=M4{j}(:,1:gopt_sn);
        Theta2{j}=Theta1{j}*(inv(Theta1{j}'*Theta1{j}))*Theta1{j}';
        Theta3{j}=(inv(Theta1{j}'*Theta1{j}))*Theta1{j}';
    end
    for ka=1:gopt_lam_idx   %ka means index for lams_eta(ka) for ka=1,,..n_lam
        if ka==1
            eta=zeros(pn*gopt_sn,1);   %%initial
            f=cell(1,pn);
            for kb=1:pn
                f{kb}=zeros(n,1);   %%initial
            end
            [eta11, f11]=algoscad(y,n,gopt_sn,pn,lams_eta(ka),G,Theta1,Theta2,Theta3,eta,f);
            eta=eta11;
            f=f11;
        end
        
        if ka>1
            [eta11, f11]=algoscad(y,n,gopt_sn,pn,lams_eta(ka),G,Theta1,Theta2,Theta3,eta,f);
            eta=eta11;
            f=f11;
        end
        %ka
    end
    opt_eta_est = reshape(eta11, [gopt_sn, pn])';
    % end of fitting the estimator
    
    %% Next we generate the estimated variance row vectors for the \theta_{ijk} based on optimal sn
    M6=cell(1,pn);
    M7=cell(1,pn);
    for j=1:pn
        M6{j}=M5{j}(1:gopt_sn);
        M7{j}=M4{j}(:,1:gopt_sn);
    end
    % end of generate the estimated variance row vectors for the \theta_{ijk}
    
    
    %% Hypotheis test
    Hnc=sort(setdiff(1:pn,Hn));
    Thetaa=cell2mat(M7);%
    Gamma=cell2mat(M6);
    hn=size(Hn,2); % hn
    G=sort(repmat(1:pn,1,gopt_sn));
    Thetaa1=cell(1,hn);      %This is Theta_{Hn} matrix
    Gamma1=cell(1,hn);      %This is Gamma_{Hn} in the form of a cell(1,hn) quantity,whose j'th elemrnent is a 1 by opt_sn vector row
    for j=1:hn
        Thetaa1{j}=Thetaa(:,G==Hn(j));
        Gamma1{j}=Gamma(:,G==Hn(j));  %% diag(sqrt(cell2mat(Gamma1)))=\hat{\Gamma}_{Hn}
    end
    Thetaa11=cell2mat(Thetaa1); % \theta_{Hn}=[E_1,..E_i,.., E_n]' is n by hn*opt_sn matrix  in matrix form, the i'th row of the matrix=Ei'
    Thetaa2=cell(1,pn-hn);   %This is Theta_{Hn^c}
    etaHnc=cell(pn-hn,1);     %This is the \hat{eta}_{Hn^c} in cell form
    for j=1:(pn-hn)
        Thetaa2{j}=Thetaa(:,G==Hnc(j));
        etaHnc{j}=eta11(G==Hnc(j),:);  %% cell2mat(etaHnc)=\hat{eta}_{Hn^c}
    end
    Thetaa22=cell2mat(Thetaa2); % \theta_{Hnc}=[F_1,..F_i,.., F_n]' is n by (pn-hn)*opt_sn matrix  in matrix form
    [MD,lamdaopt]=dantizig2(Hn,M7,n,gopt_sn,lams_d,cv_idxs_d);%MD is the hn*sn by (pn-hn)sn matrix=(\hat{w})'on page 4 of paper
    opt_lams_d(rep_ix) = lamdaopt;
end
delete(gcp('nocreate'));

[gopt_lam_d, ~, ~ ] = get_opt_lamsn(opt_lams_d, opt_lams_d);

%% 5. Get the results under best sn and lam, lam_for_dantig
% parallel runing
parpool(25);
parfor rep_ix = 1:num_rep
    %% 1.4 data
    %rep_ix = 1;
    fil_name = ['c1_' num2str(c1*1000) '_seed_' num2str(rep_ix-1)];
    fil_name
    cur_data = load([root_data_folder data_folder fil_name '.mat']);
    [n, pn, ~] = size(cur_data.X_centered);
    
    
    
    %% 2.Next, we process the X and Y
    y = cur_data.Y_centered';
    % Next we generate the f, coeffs and orthcoeffes for each realizations xij(.) on m=100 equally spaced points on [a,b]
    % i..e., get the basis (on Aug 18, 2023), in fact, M1 and M2 are no use
    M1=cell(n,pn);
    M2=cell(n,pn);
    M3=cell(n,pn); % basis
    for i=1:n
        MM1= squeeze(cur_data.X_centered(i, :, :));
        for j=1:pn
            [f,coeff,orthcoeff]=bspline2(4, sn_upper, knots, grids, MM1(j,:), An);
            M1{i,j}=f;
            M2{i,j}=coeff;
            M3{i,j}=orthcoeff;
        end
    end
    
    %% Next we generate the design matrix M4=cell(1,pn), i.e., theta
    % The output M4=cell(1,pn), where
    % M4{j}=n by sn_upper(sn_upper is number of bspline basis=100 for example) matrix,
    % similar to \theta_j except sn is changed to sn_upper
    M4=cell(1,pn); % theta, in total, note that we may not need all sn_upper theta
    MM4=cell2mat(M3);
    GM4=sort(repmat(1:pn,1, sn_upper));
    for j=1:pn
        M4{j}=centralize(MM4(:,GM4==j));
    end
    
    %% Next we generate the estimated variance row vectors for the \theta_{ijk}
    M5=cell(1,pn);
    for j=1:pn
        M5{j}=(diag(M4{j}'*M4{j}/n))';
    end
    % end of generate the estimated variance row vectors for the \theta_{ijk}
    
    
    %% Get eta estimator based on optimal sn and lam
    % The author use a loop from small to lager lams to get the good results
    G=sort(repmat(1:pn,1,gopt_sn));
    Theta1=cell(1,pn);
    Theta2=cell(1,pn);
    Theta3=cell(1,pn);
    for j=1:pn
        Theta1{j}=M4{j}(:,1:gopt_sn);
        Theta2{j}=Theta1{j}*(inv(Theta1{j}'*Theta1{j}))*Theta1{j}';
        Theta3{j}=(inv(Theta1{j}'*Theta1{j}))*Theta1{j}';
    end
    for ka=1:gopt_lam_idx   %ka means index for lams_eta(ka) for ka=1,,..n_lam
        if ka==1
            eta=zeros(pn*gopt_sn,1);   %%initial
            f=cell(1,pn);
            for kb=1:pn
                f{kb}=zeros(n,1);   %%initial
            end
            [eta11, f11]=algoscad(y,n,gopt_sn,pn,lams_eta(ka),G,Theta1,Theta2,Theta3,eta,f);
            eta=eta11;
            f=f11;
        end
        
        if ka>1
            [eta11, f11]=algoscad(y,n,gopt_sn,pn,lams_eta(ka),G,Theta1,Theta2,Theta3,eta,f);
            eta=eta11;
            f=f11;
        end
        %ka
    end
    opt_eta_est = reshape(eta11, [gopt_sn, pn])';
    % end of fitting the estimator
    
    %% Next we generate the estimated variance row vectors for the \theta_{ijk} based on optimal sn
    M6=cell(1,pn);
    M7=cell(1,pn);
    for j=1:pn
        M6{j}=M5{j}(1:gopt_sn);
        M7{j}=M4{j}(:,1:gopt_sn);
    end
    % end of generate the estimated variance row vectors for the \theta_{ijk}
    
    
    %% Hypotheis test
    %Hn = [1];
    N=10000; %we assume to have N=1000 bootstrap sample size
    %N=200000; %we assume to have N=1000 bootstrap sample size
    alpha=0.05;% we assume alpha'th quantile
    
    Hnc=sort(setdiff(1:pn,Hn));
    Thetaa=cell2mat(M7);%
    Gamma=cell2mat(M6);
    hn=size(Hn,2); % hn
    G=sort(repmat(1:pn,1,gopt_sn));
    Thetaa1=cell(1,hn);      %This is Theta_{Hn} matrix
    Gamma1=cell(1,hn);      %This is Gamma_{Hn} in the form of a cell(1,hn) quantity,whose j'th elemrnent is a 1 by opt_sn vector row
    for j=1:hn
        Thetaa1{j}=Thetaa(:,G==Hn(j));
        Gamma1{j}=Gamma(:,G==Hn(j));  %% diag(sqrt(cell2mat(Gamma1)))=\hat{\Gamma}_{Hn}
    end
    Thetaa11=cell2mat(Thetaa1); % \theta_{Hn}=[E_1,..E_i,.., E_n]' is n by hn*opt_sn matrix  in matrix form, the i'th row of the matrix=Ei'
    Thetaa2=cell(1,pn-hn);   %This is Theta_{Hn^c}
    etaHnc=cell(pn-hn,1);     %This is the \hat{eta}_{Hn^c} in cell form
    for j=1:(pn-hn)
        Thetaa2{j}=Thetaa(:,G==Hnc(j));
        etaHnc{j}=eta11(G==Hnc(j),:);  %% cell2mat(etaHnc)=\hat{eta}_{Hn^c}
    end
    Thetaa22=cell2mat(Thetaa2); % \theta_{Hnc}=[F_1,..F_i,.., F_n]' is n by (pn-hn)*opt_sn matrix  in matrix form
    MD =dantizig1(Hn,M7,n,gopt_sn,gopt_lam_d);
    S=cell(1,n); %S{i}=\hat{S}_i
    for i=1:n
        S{i}=n^(-1/2)*(inv(diag(sqrt(cell2mat(Gamma1)))))*(MD*(Thetaa22(i,:))'-(Thetaa11(i,:))')*(y(i)-Thetaa22(i,:)*cell2mat(etaHnc));
    end
    %%Next we begin wild bootstrap
    [pval, CV, TT] =mywild(alpha,N,S); [pval CV TT] %%output of the test
    
    gopt_lam = lams_eta(gopt_lam_idx);
    save_folder = ['../results/sinica_results/' data_folder];
    if ~exist(save_folder', 'dir')
        mkdir(save_folder)
    end
    
    result = struct;
    result.pval = pval;
    result.TT = TT;
    result.CV = CV;
    result.opt_eta_est = opt_eta_est;
    result.gopt_lam = gopt_lam;
    result.gopt_lam_d = gopt_lam_d;
    result.gopt_sn = gopt_sn;
    parsave([save_folder fil_name '_res.mat'],  result);
    
end
delete(gcp('nocreate'));
