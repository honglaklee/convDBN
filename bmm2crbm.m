function [CRBM, params] = bmm2crbm(X, CRBM, params)

ws = params.ws;
numhid = params.numhid;
numvis = params.numvis;
pbias = params.pbias;
sigma0 = params.sigma;

Xall = [];
patch_count= 0;
patches_per_image = round(150000/length(X)); %100;

% sample randomly from the V1 response, and then just visulaize them?
for j=1:length(X)
    imresp_v1 = X{j};
    
    for i = 1:patches_per_image,
        r = ceil(rand()*(size(imresp_v1,1)-ws+1));
        c = ceil(rand()*(size(imresp_v1,2)-ws+1));
        
        selpatch = imresp_v1(r:r+ws-1, c:c+ws-1, :);
        
        if isempty(Xall)
            Xall = zeros(numel(selpatch), length(X)*patches_per_image);
        end
        
        Xall(:, patch_count+1) = selpatch(:);
        patch_count = patch_count +1;
    end
end

% Just remove bottom 10% of the activations... Just visualize before doing
% this...?
% [sval sidx] = sort(sum(Xall,1);
idx_remove = sum(Xall,1) < quantile(sum(Xall,1), 0.25);
Xall(:, idx_remove) = [];


%% Train with Kmeans
[label, W] = litekmeans(Xall, numhid, 1, 100);

%% Train mixture of Bernoulli
fprintf('Training mixture of Bernoulli...\n');

pi = zeros(numhid,1);
for i = 1:numhid,
    pi(i) = length(find(label == i))/length(label);
end
pi = pi + 1e-3*max(pi); % prevent divide by zero
pi = pi./sum(pi);

eps = 1e-4;
W(W < eps) = eps;
W(W > 1-eps) = 1-eps; % numerical stability
W = log(W./(1-W));

% train BMM
opt_compute_ll = 1;
for t = 1:params.mmiter,
    if ~opt_compute_ll,
        fprintf('.');
    end
    
    % E step
    logQz = log(sigmoid(W))'*Xall + log(1-sigmoid(W))'*(1-Xall);
    logQz = bsxfun(@plus, log(pi), logQz);
    logQz = bsxfun(@minus, logQz, max(logQz));
    
    Qz = bsxfun(@rdivide, exp(logQz), sum(exp(logQz)));
    
    % M step
    avgRatio = bsxfun(@rdivide, Xall*Qz', sum(Qz,2)');
    avgRatio(avgRatio<eps) = eps;
    avgRatio(avgRatio>1-eps) = 1-eps;
    
    W = log(avgRatio./(1-avgRatio));
    pi = mean(Qz,2);
    
    if opt_compute_ll % compute log-likelihood
        logPk = zeros(size(Xall,2), numhid);
        parfor k=1:numhid
            wvec = W(:, k);
            logPk(:, k) = log(pi(k)) + (Xall'*log(sigmoid(wvec))+(1-Xall)'*log(1-sigmoid(wvec)));
        end
        
        maxlogPk = max(logPk,[], 2);
        logPkrel = bsxfun(@minus, logPk, maxlogPk);
        ll = mean(maxlogPk+log(sum(exp(logPkrel),2)));
        fprintf('BMM t=%g, loglik=%g\n', t, ll);
    end    
end


%%
c = mean(W,2);
W = bsxfun(@minus, W, c);
W = sigma0*W;

hbias_vec = -quantile(W'*Xall, 1-pbias, 2);

hidprob = sigmoid(1/sigma0.*bsxfun(@plus, W'*Xall, hbias_vec));

xmarginal = mean(Xall,2);
c = sigma0*log(xmarginal./(1-xmarginal)) - mean(W*hidprob,2);

vbias_vec = c;

W = reshape(W, [ws^2, numvis, numhid]);

vbias_vec = reshape(vbias_vec, [ws^2, numvis]);
vbias_vec = mean(vbias_vec, 1)';

W = W./10;
hbias_vec = hbias_vec/10;

CRBM.W = reshape(W,[ws,ws,numvis,numhid]);
CRBM.hbias = hbias_vec;
CRBM.vbias = vbias_vec;
params.sigma = sigma0;

return

function [label,center] = litekmeans(X, k, opt_verbose, MAX_ITERS)

if ~exist('opt_verbose', 'var')
    opt_verbose = false;
end

if ~exist('MAX_ITERS', 'var')
    MAX_ITERS = 50;
end

n = size(X,2);
last = 0;
label = ceil(k*rand(1,n));  % random initialization
itr=0;
% MAX_ITERS=50;
while any(label ~= last)
    itr = itr+1;
    if opt_verbose
        fprintf(1, '%d(%d)..', itr, sum(label ~= last));
    end
    
    E = sparse(1:n,label,1,n,k,n);  % transform label into indicator matrix
    center = X*(E*spdiags(1./sum(E,1)',0,k,k));    % compute center of each cluster
    last = label;
    [val,label] = max(bsxfun(@minus,center'*X,0.5*sum(center.^2,1)')); % assign samples to the nearest centers
    if (itr >= MAX_ITERS) break; end;
end

if opt_verbose
    fprintf(1,'\n');
end
