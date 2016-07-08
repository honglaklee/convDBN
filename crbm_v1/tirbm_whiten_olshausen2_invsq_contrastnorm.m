function im_out = tirbm_whiten_olshausen2_invsq_contrastnorm(im, Qnn, D, noenhance)

global global_Qss_freq global_Qnn global_filt_orig; % Use global variable to speed up the the code
opt_global = true;

if ~exist('D', 'var'), D = 16; end

if size(im,3)>1, im = rgb2gray(im); end
im = double(im);

im = im - mean(im(:));
im = im./std(im(:));

N1 = size(im, 1);
N2 = size(im, 2);

% [fx fy]=meshgrid(-N1/2:N1/2-1, -N2/2:N2/2-1);
% rho=sqrt(fx.*fx+fy.*fy)';

% f_0=0.4*mean([N1,N2]);
% filt=rho.*exp(-(rho/f_0).^4);
% filt=rho./(1+5*(rho/f_0).^2);

if ~opt_global
    % load Qss_kyoto.mat Qss Qss_freq
    load Qss_kyoto.mat Qss_freq

    filt_new = (sqrt(Qss_freq)./(Qss_freq+Qnn));
else
    if isempty(global_Qss_freq)
        load Qss_kyoto.mat Qss_freq
        global_Qss_freq = Qss_freq;
    end
    Qss_freq = global_Qss_freq;
    
    if isempty(global_Qnn) || Qnn ~= global_Qnn
        global_Qnn = Qnn;
        global_filt_orig = (sqrt(Qss_freq)./(Qss_freq+Qnn));
    end
    filt_new = global_filt_orig;
end

% Qss_freq = abs(fftshift(fft2(Qss)));

% load Qss_kyoto.mat Qss_freq
% global_filt_orig = (sqrt(Qss_freq)./(Qss_freq+Qnn));
filt_new = imresize(filt_new, [N1, N2], 'bicubic');

If=fft2(im);
imw=real(ifft2(If.*fftshift(filt_new)));

% contrast normalization
[x y] = meshgrid(-D/2:D/2);
G = exp(-0.5*((x.^2+y.^2)/(D/2)^2));
G = G/sum(G(:));
imv = conv2(imw.^2,G,'same');
imv2 = conv2(ones(size(imw)),G,'same');
% imn = imw./sqrt(imv);
if ~noenhance
    imn = imw; % This is modified version (no contrast normalization)
%     imn = imw./sqrt(imv);
%     imn = imw./sqrt(imv).*sqrt(imv2);
else
    cutoff = quantile(sqrt(imv(:)), 0.3); 
    imn = imw./max(sqrt(imv), cutoff);
%     cutoff = quantile(sqrt(imv(:)./imv2(:)), 0.3); 
%     imn = imw./max(sqrt(imv)./sqrt(imv2), cutoff);
end

im_out = imn/std(imn(:)); % 0.1 is the same factor as in make-your-own-images

return


%
if 0
    
clear

load ~/robo/visionnew/trunk/sparsenet/data/IMAGES.mat
load ~/robo/visionnew/trunk/sparsenet/data/IMAGES_RAW.mat

fpath = '~/robo/brain/data/kyoto/gray8bit';
flist = dir(sprintf('%s/*.png', fpath));

Qss = 0;
Qss_array = [];
for i= 1:length(flist);
    fprintf('.');
    im = imread(sprintf('%s/%s', fpath, flist(i).name));
    im = imresize(im, 0.5, 'bicubic');
    im = double(im);
    im = im-mean(im(:));
    im = im./std(im(:));

    f1 = ifftshift(ifft2(fft2(fftshift(im)).*fft2(fftshift(fliplr(flipud(im))))));

    Qss_array(:,:,i) = f1;
    Qss = Qss+f1;
    if 0
        figure(1), imagesc(im), colormap gray
        figure(2), imagesc(f1);
    end
end

Qss = Qss/max(Qss(:));
figure(1), subplot(2,1,1), plot(mean(Qss,1)); subplot(2,1,2), plot(mean(Qss,2));
figure(2), imagesc(Qss)

% figure(4), display_images(Qss_array), colormap gray
% figure(5), display_images(IMAGESr), colormap gray
%%
% rotational filter
[r0 c0] = find(Qss == max(Qss(:)));
[X, Y] = meshgrid(1:size(im,1), 1:size(im,2));
X = X - r0;
Y = Y - c0;
R = sqrt(X.^2 + Y.^2);
% R

figure(2), plot(R(:), Qss(:), '.')

%% Learning optimal filter
% temp = abs(fftshift(fft(Qss(:, end/2))));
filt_new = abs(fftshift(fft2(Qss)));
% filt_new = real(fftshift(fft2(Qss, 1000, 1000)));
filt_new = (sqrt(filt_new)./(filt_new+1));


% Olshausen's method
im = IMAGESr(:,:,3);

N1 = size(im,1);
N2 = size(im,2);

[fx fy]=meshgrid(-N1/2:N1/2-1, -N2/2:N2/2-1);
rho=sqrt(fx.*fx+fy.*fy)';
f_0=0.4*mean([N1,N2]);
filt=rho.*exp(-(rho/f_0).^2);

imagew=real(ifft2(fft2(im).*fftshift(filt)));
figure(10), imagesc(imagew), colormap gray

% filt_new=rho./(1+10*(rho/f_0).^2);
imagew_new=real(ifft2(fft2(im).*fftshift(imresize(filt_new, size(im), 'bicubic'))));
figure(11), imagesc(imagew_new), colormap gray
figure(12), plot(filt_new(end/2,:))
figure(13), imagesc(filt_new);

figure(21), imagesc(im), colormap gray
figure(22), plot(filt(end/2,:))

Qss_new = ifftshift(ifft2(fft2(fftshift(imagew_new)).*fft2(fftshift(fliplr(flipud(imagew_new))))));
Qss_olshausen = ifftshift(ifft2(fft2(fftshift(imagew)).*fft2(fftshift(fliplr(flipud(imagew))))));

Identity = zeros(size(Qss_new));
[v ix] = max(Qss_new(:));
Identity(ix) = 1;
norm(Qss_new(:)./norm(Qss_new(:)) - Identity(:))
norm(Qss_olshausen(:)./norm(Qss_olshausen(:)) - Identity(:))

end
