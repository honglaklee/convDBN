function [H HP Hc HPc imdata_v0] = crbm_v1_response(im, CRBM, sigma, spacing, imsize, D, ws_pad, noiselevel)
%
if ~exist('sigma','var') || isempty(sigma), 
    sigma = 1; 
end
if ~exist('noiselevel', 'var'), 
    noiselevel = 0.5; 
end

%%% image preprocessing
if size(im,3)>1, im2 = double(rgb2gray(im));
else im2 = double(im); end

if ws_pad,
    padval = (mean(mean(im2(:, [1,size(im2,2)]))) + mean(mean(im2([1,size(im2,1)],:))))/2;
    im2 = padarray(im2, [ws_pad, ws_pad], padval);
end

imsize_max = 180;
ratio = min([imsize_max/size(im,1), imsize_max/size(im,2), 1]);
if size(im,1)*ratio < imsize || size(im,2)*ratio < imsize,
    ratio = max([imsize/size(im,1), imsize/size(im,2)]);
end
im2 = imresize(im2, [round(ratio*size(im,1)), round(ratio*size(im,2))], 'bicubic');

im2_new = tirbm_whiten_olshausen2_invsq_contrastnorm(im2, noiselevel, D, true);

im2 = im2_new;
im2 = im2-mean(mean(im2));
im2 = im2/sqrt(mean(mean(im2.^2)));

if ndims(CRBM.W) == 3,
    ws = sqrt(size(CRBM.W, 1));
elseif ndims(CRBM.W) == 4,
    ws = size(CRBM.W, 1);
end

im2 = trim_image(im2, ws, spacing);
imdata_v0 = im2/1.5;

%%% compute response
[H HP Hc HPc] = crbm_inference_response(imdata_v0, CRBM, sigma, spacing);

return

function im2 = trim_image(im2, ws, spacing)
% % Trim image so that it matches the spacing.
if mod(size(im2,1)-ws+1, spacing)~=0
    n = mod(size(im2,1)-ws+1, spacing);
    im2(1:floor(n/2), : ,:) = [];
    im2(end-ceil(n/2)+1:end, : ,:) = [];
end
if mod(size(im2,2)-ws+1, spacing)~=0
    n = mod(size(im2,2)-ws+1, spacing);
    im2(:, 1:floor(n/2), :) = [];
    im2(:, end-ceil(n/2)+1:end, :) = [];
end
return
