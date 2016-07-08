function [CRBM, params] = gmm2crbm(X,CRBM,params)
%   Ey: (1/std^2)*[1/2*v'v - v'Wh - b'h - c'v]

n = 300000; % number of patches
K = params.numhid;
ws = params.ws;

fea_all = zeros(ws^2,n);
k = 0;
for i = 1:length(X),
    pperimg = ceil(n/length(X));
    curimg = X{i};
    for j = 1:pperimg,
        k = k + 1;
        if k > n, break; end
        rowidx = randi(size(curimg,1)-ws);
        colidx = randi(size(curimg,2)-ws);
        fea_all(:,k) = vec(curimg(rowidx:rowidx+ws-1,colidx:colidx+ws-1));
    end
end

% KMeans to initialize GMM
addpath ../KMeanstoolbox/;
fea_all = double(fea_all);
[label, center] = litekmeans(fea_all', K, true, 250);
center = center';
params.sigma = sqrt(mean(mean( (fea_all - center(:, label)).^2, 2)));

% initialize GMMs
E = sparse(1:n,label,1,n,K,n)';  % transform label into indicator matrix

p0 = mean(E,2);
p0 = p0 + 10^-8;
p0 = p0/sum(p0);
p0 = sparse(p0);
mu0 = center;
sigma0 = sqrt(mean(mean( (fea_all - center(:, label)).^2, 2))); % Using sigma0*I in this example

% GMM iterations
fea_all = double(fea_all);
for t = 1:params.mmiter,
    fprintf('GMM iteration: %g...\n', t);
    
    % E step
    distsq = bsxfun(@minus,mu0'*fea_all, 0.5*sum(mu0.^2,1)');
    distsq = bsxfun(@minus,distsq, 0.5*sum(fea_all.^2,1));
    
    ll = log(mean(sum(exp(bsxfun(@plus, distsq./2/sigma0^2, log(p0))))));
    
    distsq = bsxfun(@minus, distsq, max(distsq)); % this for a numerical stability
    distsq = exp(bsxfun(@plus, distsq./2/sigma0^2, log(p0)));
    distsq = bsxfun(@rdivide, distsq, sum(distsq)); % softmax

    fprintf('iteration %g: ll (relative): %g...\n', t, ll);    
    
    % M step
    p0 = mean(distsq,2);
    mu0 = zeros(size(mu0));
    errvecsq_total = 0;
    z_total = 0;
    for k = 1:K
        z_k = distsq(k,:)';
        mu_k = fea_all*z_k/sum(z_k);
        mu0(:,k) = mu_k;

        errvecsq_total = errvecsq_total + bsxfun(@minus, fea_all, mu_k).^2*z_k;
        z_total = z_total + sum(z_k);
    end
    sigma0 = sqrt(mean(errvecsq_total/sum(z_total)));
end

% initialize RBM parameters
GMM.p0 = p0;
GMM.mu = mu0;
GMM.sigma = sigma0;

c = GMM.mu*GMM.p0;
sigmagmm = GMM.sigma;
Wgmm = bsxfun(@minus, GMM.mu, c);
bgmm = zeros(length(GMM.p0),1);

% heuristic..
for k = 1:length(GMM.p0)
    bgmm(k) = sigmagmm^2*log(GMM.p0(k)*K)-GMM.mu(:,k)'*GMM.mu(:,k)/2;
end
bgmm = bgmm(:);

Wgmm = permute(reshape(Wgmm,[params.numvis,ws^2,K]),[2 1 3]);
c = reshape(c,[params.numvis,ws^2]);
c = sum(c,2)/ws^2;

% initialize CRBM parameter
CRBM.W = reshape(Wgmm,[ws,ws,params.numvis,K]);
CRBM.hbias = bgmm;
CRBM.vbias = c;
params.sigma = sigmagmm;

return;