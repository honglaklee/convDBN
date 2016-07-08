function display_crbm_v2_bases(W, V1, expandfactor, opt_nonneg, cols)
addpath /mnt/neocortex/scratch/kihyuks/library/Display_Networks/;

if ~exist('opt_nonneg', 'var'), opt_nonneg = false; end
if ~exist('expandfactor', 'var'), expandfactor = 4; end
if ndims(W) == 4,
    W = reshape(W,size(W,1)*size(W,2),size(W,3),size(W,4));
end
W1 = V1.W;
if ndims(W1) == 4,
    W1 = reshape(W1,size(W1,1)*size(W1,2),size(W1,3),size(W1,4));
end
W = double(W);
W1 = double(W1);

images = [];
ws = sqrt(size(W,1));
for i=1:size(W, 3)
    poshid_reduced = reshape(W(:,:,i), [ws, ws, size(W,2)]);
    poshid_reduced(poshid_reduced<0) = 0; % make the coefficients truncated
    
    poshid_expand = imresize(double(poshid_reduced), expandfactor, 'bicubic');    
    negdata_expand = crbm_reconstruct_LB_fixconv(poshid_expand, W1);

    if isempty(images), images = zeros(size(negdata_expand,1), size(negdata_expand,2), size(W,3)); end
    images(:,:,i) = negdata_expand;
end

if exist('cols', 'var')
    display_network(reshape(images, size(images,1)*size(images,2), size(images,3)), true, true, cols, false);
else
    display_network(reshape(images, size(images,1)*size(images,2), size(images,3)), true, true);
end

return

function negdata2 = crbm_reconstruct_LB_fixconv(S, W)

ws = sqrt(size(W,1));
patch_M = size(S,1);
patch_N = size(S,2);
numchannels = size(W,2);
numbases = size(W,3);

% Note: Reconstruction was off by a few pixels in the original code (above
% versions).. I fixed this as below:
S2 = zeros(size(S));
S2(2:end,2:end,:) = S(1:end-1,1:end-1,:);
negdata2 = zeros(patch_M, patch_N, numchannels);
if numchannels == 1
    H = reshape(W,[ws,ws,numbases]);
    negdata2 = sum(conv2_mult_pairwise(S2, H, 'same'),3);
else
    for b = 1:numbases,
        H = reshape(W(:,:,b),[ws,ws,numchannels]);
        negdata2 = negdata2 + conv2_mult(S2(:,:,b), H, 'same');
    end
end

return

function y = conv2_mult(a, B, convopt)
y = [];
if size(a,3)>1
    for i=1:size(a,3)
        y(:,:,i) = conv2(a(:,:,i), B, convopt);
    end
else
    for i=1:size(B,3)
        y(:,:,i) = conv2(a, B(:,:,i), convopt);
    end
end
return

function y = conv2_mult_pairwise(a, B, convopt)
y = [];

assert(size(a,3)==size(B,3));
for i=1:size(a,3)
    y(:,:,i) = conv2(a(:,:,i), B(:,:,i), convopt);
end

return
