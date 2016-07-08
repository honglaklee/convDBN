%%% demo for convolutional deep belief network
%%% with 2 layers of convolutional restricted Boltzmann machine with
%%% probabilistic max-pooling

function demo_cdbn(objclass, spacing_V1, pbias_V1, plambda_V1, numhid_V1, l2reg_V1, spacing_V2, pbias_V2, plambda_V2, numhid_V2, l2reg_V2)

% parameters for the first layer
if ~exist('spacing_V1','var'), spacing_V1 = 2; end
if ~exist('pbias_V1','var'), pbias_V1 = 0.002; end
if ~exist('plambda_V1','var'), plambda_V1 = 5; end
if ~exist('numhid_V1','var'), numhid_V1 = 24; end
if ~exist('l2reg_V1','var'), l2reg_V1 = 0.01; end

% object class ('Faces_easy', 'car_side')
if ~exist('objclass','var'), objclass = 'Faces_easy'; end

% parameters for the second layer
if ~exist('spacing_V2','var'), spacing_V2 = 3; end
if ~exist('pbias_V2','var'), pbias_V2 = 0.003; end
if ~exist('plambda_V2','var'), plambda_V2 = 5; end
if ~exist('numhid_V2','var'), numhid_V2 = 40; end
if ~exist('l2reg_V2','var'), l2reg_V2 = 0.02; end


%% 1st layer (natural images)
dataname = 'olshausen';
fname_V1 = sprintf('crbm_V1_%s_b%02d_pb%g_pl%g_l2r%g_sp%d',dataname,numhid_V1,pbias_V1,plambda_V1,l2reg_V1,spacing_V1);
try
    load(sprintf('pretrain/%s.mat',fname_V1),'CRBM','params','CDBN');    
catch
    load data/olshausen_single.mat;
    X = images_all;
    
    % train real-binary crbm
    [CRBM, params, CDBN] = crbm_train(X,struct('sigma',0.2,'verbose',1,'batchsize',100,'batch_ws',70,'epsilon',2e-2,'intype','real','nlayer',1,'dataSet',dataname,...
        'spacing',spacing_V1,'pbias',pbias_V1,'plambda',plambda_V1,'numhid',numhid_V1,'l2reg',l2reg_V1,'l1reg',0,'maxiter',500,'savedir','results','optdouble',1,'ws',10,'showfig',1));
    % save
    if ~exist('pretrain','dir'),
        mkdir('pretrain');
    end
    save(sprintf('pretrain/%s.mat',fname_V1),'CRBM','params','CDBN');
end


%% 2nd layer (caltech 101)
%%% compute first layer response
addpath crbm_v1;
spacing = CDBN{1}.params.spacing;
H = compute_v1_response(objclass, CRBM, params, spacing, 60);

%%% determine maximum input size
batch_ws = inf;
for i = 1:length(H),
    cur_batch_ws = min(size(H{i},1),size(H{i},2));
    if cur_batch_ws >= 30,
        batch_ws = min(batch_ws,cur_batch_ws);
    end
end
batch_ws = min(batch_ws,50);

fname_V2 = sprintf('crbm_V2_%s_b%02d_pb%g_pl%g_l2r%g_sp%d_b%02d_pb%g_pl%g_l2r%g_sp%d',objclass,numhid_V1,pbias_V1,plambda_V1,l2reg_V1,CDBN{1}.params.spacing,numhid_V2,pbias_V2,plambda_V2,l2reg_V2,spacing_V2);
% train crbm V2
[CRBM, params, CDBN] = crbm_train(H,struct('batch_ws',batch_ws,'nlayer',2,'ws',12,'epsilon',0.01,'batchsize',100,'dataSet',objclass,...
    'spacing',spacing_V2,'pbias',pbias_V2,'plambda',plambda_V2,'numhid',numhid_V2,'l2reg',l2reg_V2,'l1reg',1e-6,'maxiter',100,'savedir','results','verbose',1,'sigma',0.2,'sigma_stop',0.2,'eta_sparsity',0.01,'batchperiter',2), CDBN);

save(sprintf('pretrain/%s.mat',fname_V2),'CRBM','params','CDBN');

