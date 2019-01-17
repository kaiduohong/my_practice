
 %every sample per row

load('Caltech10_SURF_L10.mat');%source domain

fts=fts./repmat(sum(fts,2),1,size(fts,2));  %average every row

Xs=zscore(fts,1);clear fts



Ys=labels;clear labels
load('amazon_SURF_L10.mat');%targetdomain
fts=fts./ repmat(sum(fts,2),1,size(fts,2));
Xt=zscore(fts,1);
clear fts
Yt=labels;
clear labels

options.T = 50; % #iterations , default=10
options.gamma = 2; % the parameter for kernel
options.kernel_type = 'linear';
options.lambda = 1.0 ;
options.dim = 25;
[ Acc , Acc_iter ,A] = MyJDA(Xs , Ys , Xt , Yt , options ) ;
disp (Acc) ;
