function[acc,acc_ite,A]=MyJDA(X_src,Y_src,X_tar,Y_tar,options)
%ThisistheimplementationofJointDistributionAdaptation.
%Reference:MingshengLongetal.Transferfeaturelearningwithjointdistribution
%adaptation.ICCV2013.
%Inputs:
%%%X_src:sourcefeaturematrix,ns*n_feature
%%%Y_src:sourcelabelvector,ns*1
%%X_tar:targetfeaturematrix,nt*n_feature
%%%Y_tar:targetlabelvector,nt*1
%%%options:optionstruct
%%%%%lambda:regularizationparameter
%%%%%dim:dimension after adaptation,dim<=n_feature
%%%%%kernel_tpye:kernelname,choosefrom'primal'|'linear'|'rbf'
%%%%%gamma:bandwidthforrbfkernel,canbemissedforotherkernels
%%%%%T:n_iterations,T>=1.T<=10issuffice
%Outputs:
%%%acc:finalaccuracyusingknn,float
%%%acc_ite:listofallaccuraciesduringiterations
%%%A:finaladaptationmatrix,(ns+nt)*(ns+nt)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%Setoptions
lambda=options.lambda;  
dim=options.dim;  
kernel_type=options.kernel_type;
gamma=options.gamma;
T=options.T;
acc_ite=[];
Y_tar_pseudo=[];
%%Iteration
for i = 1:T  %t times
    % Z = A'X',the transformed result
    [Z,A]=JDA_core(X_src,Y_src,X_tar,Y_tar_pseudo,options);
    %normalization for better classification performance
    Z=Z*diag(sparse(1./sqrt(sum(Z.^2))));

    Zs=Z(:,1:size(X_src,1));
    Zt=Z(:,size(X_src,1)+1:end);
    knn_model = fitcknn(Zs',Y_src,'NumNeighbors',5);
    Y_tar_pseudo = knn_model.predict(Zt');
    acc=length(find(Y_tar_pseudo == Y_tar))/length(Y_tar);
    fprintf('JDA+NN=%0.4f\n',acc);
    acc_ite=[acc_ite;acc];
end

end

function[Z,A]=JDA_core(X_src,Y_src,X_tar,Y_tar_pseudo,options)
%%Setoptions
lambda=options.lambda;%%lambdafortheregularization
dim=options.dim;%%dimisthedimensionafteradaptation,dim<=m
kernel_type=options.kernel_type;%%kernel_typeisthekernelname,primal|linear|rbf
gamma=options.gamma;%%gammaisthebandwidthofrbfkernel
%%ConstructMMDmatrix
X=[X_src',X_tar'];  %one sample per column
X=X*diag(sparse(1./sqrt(sum(X.^2))));  %normalize every column
[m,n]=size(X);  %n samples n = ns + nt
ns=size(X_src,1);
nt=size(X_tar,1);
e=[1/ns*ones(ns,1);-1/nt*ones(nt,1)]; 
C=length(unique(Y_src));  %class number

%%%M0  why it should multiple by C
M=e*e'*C;%multiply C for better  normalization
%%%Mc
N=0; %formular 6.16
if~isempty(Y_tar_pseudo)&&length(Y_tar_pseudo) == nt
    for c = reshape(unique(Y_src),1,C)
        e=zeros(n,1);
        e(Y_src==c)=1/length(find(Y_src==c));
        e(ns+find(Y_tar_pseudo==c))=-1/length(find(Y_tar_pseudo==c));
        e(isinf(e))=0;
        N=N+e*e';

    end
end

M=M+N;
M=M/norm(M,'fro'); 

%%CenteringmatrixH
H = eye(n) - 1/n*ones(n,n);
%%Calculation


if strcmp(kernel_type,'primal')
    [A,~]=eigs(X*M*X'+lambda*eye(m),X*H*X',dim,'SM');%estimate (X*M*X'+lambda*eye(m))A = X*H*X'A \Phi
    Z=A'*X;
else
    K=kernel_jda(kernel_type,X,[],gamma);
  
    
    [A,VVV]=eigs(K*M*K'+lambda*eye(n),K*H*K',dim,'SM');
    disp(VVV)
    disp('KKKKKKKKKKKKK')
    disp(K(1:10,1:10))
    disp('AAAAAAAAAA')
    disp(A(1:10,1:10))
    Z=A'*K;
    disp('ZZZZZZZZZZZ')
    disp(Z(1:10,1:10))
    pause;
    
end

end
%WithFastComputationoftheRBFkernelmatrix
%Tospeedupthecomputation,weexploitadecompositionoftheEuclideandistance(norm)
%
%Inputs:
%ker:'linear','rbf','sam'
%X:datamatrix(features*samples)
%gamma:bandwidthoftheRBF/SAMkernel
%Output:
%K:kernelmatrix
%
%GustavoCamps?Valls
%2006(c)
%Jordi(jordi@uv.es),2007
%2007?11:if/then?>switch,andfixedRBFkernel
%ModifiedbyMingshengLong
%2013(c)
%MingshengLong(longmingsheng@gmail.com),2013
function K=kernel_jda(ker,X,X2,gamma)
switch ker
case 'linear'

if isempty(X2)
    K=X'*X;
else
    K=X'*X2;
end
case 'rbf'
    n1sq=sum(X.^2,1);
    n1=size(X,2);
if isempty(X2)
    D=(ones(n1,1)*n1sq)'+ones(n1,1)*n1sq-2*X'*X;
else
    n2sq=sum(X2.^2,1);
    n2=size(X2,2);
    D=(ones(n2,1)*n1sq)'+ones(n1,1)*n2sq-2*X'*X2;
end
K=exp(-gamma*D);

case 'sam'

if isempty(X2)
    D=X'*X;
else
    D=X'*X2;
end
    K=exp(-gamma*acos(D).^2);
   otherwise
    error(['Unsupportedkernel' ker])
end
end