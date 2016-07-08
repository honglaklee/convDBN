function [CRBM, params ,CDBN] = crbm_train(X,params,CDBN)
%% convolutional RBM
%   Ey: (1/std^2)*[v'v - v'Wh - b'h - c'v]

addpath utils/;

if ~exist('CDBN','var'),
    CDBN = cell(1,1);
end

if ~exist('params','var'),
    params = struct;
end

%%% --- set up hyper parameters --- %%%
params = makeCRBMparams(params);
params.numvis = size(X{1},3);
if isempty(params.batch_ws),
    params.batch_ws = inf;
    for i = 1:length(X),
        params.batch_ws = min(params.batch_ws,min(size(X{i},1),size(X{i},2)));
    end
end
params.batch_ws = params.batch_ws - rem(params.batch_ws - (params.ws + params.spacing - 1), params.spacing);
if isempty(params.batchsize),
    params.batchsize = length(X);
end
if ~isfield(params,'optmeannorm'),
    if strcmp(params.intype,'real'),
        params.optmeannorm = 1;
    else
        params.optmeannorm = 0;
    end
end
if ~isfield(params,'batchperiter'),
    params.batchperiter = 1;
end

initialmomentum  = 0.5;
finalmomentum    = 0.9;


%%% --- initialize model parameters --- %%%
randn('state',0);
rand('state',0);

% initialize crbm weights
if params.optmminit,
    if strcmp(params.intype,'real'),
        [CRBM, params] = gmm2crbm(X,[],params);
        params.sigma_schedule = 0;
    elseif strcmp(params.intype,'binary'),
        if isfield(params,'sigma'),
            params.sigma_schedule = 1;
            if ~isfield(params,'sigma_stop'),
                params.sigma_stop = params.sigma/2;
            end
        else
            params.sigma = 1;
        end
        [CRBM, params] = bmm2crbm(X,[],params);
    else
        params.optmminit = 0;
    end
end

if ~params.optmminit,
    CRBM.W = 0.01*randn(params.ws, params.ws, params.numvis, params.numhid);
    CRBM.hbias = zeros(params.numhid, 1);
    CRBM.vbias = zeros(params.numvis, 1);
    if isfield(params,'sigma'),
        params.sigma_schedule = 1;
        if ~isfield(params,'sigma_stop'),
            params.sigma_stop = params.sigma/2;
        end
    else
        if strcmp(params.intype,'real'),
            [~, params] = gmm2crbm(X,[],params);
        elseif strcmp(params.intype,'binary'),
            params.sigma = 1;
        end
    end
end

%%% --- other variables --- %%%
% intermediate variables during training
PAR.Winc = zeros(size(CRBM.W));
PAR.hbiasinc = zeros(size(CRBM.hbias));
PAR.vbiasinc = zeros(size(CRBM.vbias));

params.maxiter = params.maxiter;

% monitoring variables
error_history = zeros(params.maxiter,1);
sparsity_history = zeros(params.maxiter,1);
sigma_history = zeros(params.maxiter,1);
error_for_sigma_history = zeros(params.maxiter,1);
PAR.runningavg_prob = [];

% other parameters
opt.visrow = params.batch_ws;
opt.viscol = params.batch_ws;
opt.hidrow = opt.visrow - params.ws + 1;
opt.hidcol = opt.viscol - params.ws + 1;
opt.vissize = opt.visrow*opt.viscol;
opt.hidsize = opt.hidrow*opt.hidcol;

% filename to save
if isempty(params.fname_save),
    params.fname_save = sprintf('crbm_%s_%s_ws%d_v%d_h%d_init%d_l2reg%g_eps%g_pb%g_pl%g_%s_date_%s', params.intype, params.dataset, params.ws, params.numvis, params.numhid, params.optmminit, params.l2reg, params.epsilon, params.pbias, params.plambda, params.sptype, datestr(now, 30));
end
fname_mat = sprintf('%s/%s.mat', params.savedir, params.fname_save);
fname_png = sprintf('%s/%s.png', params.savedir, params.fname_save);

disp(params);


%%% --- train (real, binary)-binary convolutional RBM --- %%%

t20S = tic;
for t = 1:params.maxiter,
    recon_err_epoch = zeros(params.batchsize,1);
    sparsity_epoch = zeros(params.batchsize,1);
    recon_err_for_sigma_epoch = zeros(params.numvis,1);
    
    epsilon = params.epsilon/(1+params.epsdecay*t);
    randidx = randsample(length(X),params.batchsize,length(X) < params.batchsize);
    
    tS = tic;
    for b = 1:(params.batchsize/params.batchperiter),
        for bi = 1:params.batchperiter,
            %%% prepare large image for the current batch of training
            Xb = trim_image_square(X{randidx(b)},params.ws,params.batch_ws,params.spacing);
            if params.optmeannorm,
                % mean normalization
                Xb = Xb - mean(Xb(:));
                % flip the image
                if rand() > 0.5,
                    Xb = fliplr(Xb);
                end
            end
            PAR.vis = Xb;            
            
            %%% compute gradient of log-likelihood (contrastive divergence)
            [CRBM, PAR] = fobj_crbm(CRBM, PAR, params, opt);
            
            %%% compute gradient of sparsity penalty
            [CRBM, PAR] = fobj_sparsity(CRBM, PAR, params, opt);
            
            recon_err_epoch(b) = PAR.ferr;
            sparsity_epoch(b) = PAR.sparsity*params.batchperiter;
            recon_err_for_sigma_epoch = recon_err_for_sigma_epoch + double(PAR.recon_err);
            
            if t < 5,
                momentum = initialmomentum;
            else
                momentum = finalmomentum;
            end
            
            %%% update model parameters
            PAR.Winc = momentum*PAR.Winc + epsilon*PAR.dW_total;
            PAR.hbiasinc = momentum*PAR.hbiasinc + epsilon*PAR.dh_total;
            PAR.vbiasinc = momentum*PAR.vbiasinc + epsilon*PAR.dv_total;
            
            CRBM.W = CRBM.W + PAR.Winc;
            CRBM.hbias = CRBM.hbias + PAR.hbiasinc;
            if ~params.optmeannorm,
                CRBM.vbias = CRBM.vbias + PAR.vbiasinc;
            end
        end
    end
    error_history(t) = double(mean(recon_err_epoch));
    sparsity_history(t) = double(mean(sparsity_epoch));
    sigma_history(t) = double(params.sigma);
    sigma_recon = sqrt(mean(recon_err_for_sigma_epoch/b));
    error_for_sigma_history(t) = sigma_recon;
    
    tE = toc(tS);
    if params.verbose,
        fprintf('epoch %d: error=%.5g, sparsity=%.5g, sigma=%.5g, time=%.5g\n', t, double(error_history(t)), double(sparsity_history(t)), params.sigma, tE);
        if params.nlayer == 1,
            display_network_nonsquare(reshape(CRBM.W,params.ws^2,params.numhid));
        elseif params.nlayer == 2,
            display_crbm_v2_bases(CRBM.W, CDBN{1}, CDBN{1}.params.spacing);
        end
    end
    
    %%% update CDBN
    CDBN{params.nlayer}.W = double(CRBM.W);
    CDBN{params.nlayer}.hbias = double(CRBM.hbias);
    CDBN{params.nlayer}.vbias = double(CRBM.vbias);
    CDBN{params.nlayer}.params = params;
    
    if mod(t, 20) == 0,
        t20E = toc(t20S);
        fprintf('epoch %d: error=%.5g, sparsity=%.5g, sigma=%.5g, time=%.5g\n', t, double(error_history(t)), double(sparsity_history(t)), params.sigma, t20E);
        t20S = tic;
        if params.showfig,
            figure(1);
            if params.nlayer == 1, 
                display_network_nonsquare(reshape(CRBM.W,params.ws^2,params.numhid));
            elseif params.nlayer == 2, 
                display_crbm_v2_bases(CRBM.W, CDBN{1}, CDBN{1}.params.spacing);
            end
            figure(2),
            subplot(2,1,1), plot(error_history(1:t)); title('reconstruction error');
            subplot(2,1,2), plot(sparsity_history(1:t)); title('sparsity');
        end
    end
    
    if strcmp(params.intype,'real'),
        if params.sigma_schedule,
            if params.sigma > params.sigma_stop,
                params.sigma = params.sigma*0.99;
            end
        else
            params.sigma = (1-params.eta_sigma)*params.sigma + params.eta_sigma*sqrt(sigma_recon);
        end
    end
    
    % save parameters
    save_vars(fname_mat, CRBM, CDBN, params, error_history, sparsity_history);
%     save_visualization(CRBM, CDBN, params, t);
end

%% visualization
figure(1);
if params.nlayer == 1,
    display_network_nonsquare(reshape(CRBM.W,params.ws^2,params.numhid));
elseif params.nlayer == 2,
    display_crbm_v2_bases(CRBM.W, CDBN{1}, CDBN{1}.params.spacing);
end
saveas(gcf, fname_png);

figure(2),
subplot(2,1,1), plot(error_history(1:t)); title('reconstruction error');
subplot(2,1,2), plot(sparsity_history(1:t)); title('sparsity');

CRBM = save_vars(fname_mat,CRBM,CDBN,params,error_history,sparsity_history);

return;

function CRBM = save_vars(fname_mat,CRBM,CDBN,params,error_history,sparsity_history)

CRBM.W = double(CRBM.W);
CRBM.hbias = double(CRBM.hbias);
CRBM.vbias = double(CRBM.vbias);

CRBM = rmfield(CRBM,'Wlr');
CRBM = rmfield(CRBM,'vbiasmat');
CRBM = rmfield(CRBM,'hbiasmat');

save(fname_mat,'CRBM','CDBN','params','error_history','sparsity_history');
return;

function save_visualization(CRBM, CDBN, params, t)
if ~exist('visualization','dir'),
    mkdir('visualization');
end
if ~exist(sprintf('visualization/%s',params.fname_save),'dir'),
    mkdir(sprintf('visualization/%s',params.fname_save));
end

fig = figure;
fig_save = sprintf('visualization/%s/%04d.jpg',params.fname_save,t);
if params.nlayer == 1,
    display_network_nonsquare(reshape(CRBM.W,params.ws^2,params.numhid));
elseif params.nlayer == 2,
    display_crbm_v2_bases(CRBM.W, CDBN{1}, CDBN{1}.params.spacing);
end
print(fig,'-djpeg',fig_save);
close(fig);

return;
