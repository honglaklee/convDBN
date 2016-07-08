function h=display_network_nonsquare(A, m, numcols, figstart)
%  display_network -- displays the state of the network
%  A = basis function matrix

warning off all

if exist('figstart', 'var') && ~isempty(figstart), figure(figstart); end

[L M]=size(A);
if ~exist('numcols', 'var')
    numcols = ceil(sqrt(L));
    while mod(L, numcols), numcols= numcols+1; end
end
ysz = numcols;
xsz = ceil(L/ysz);

if ~exist('m','var'),
    m=floor(sqrt(M*ysz/xsz));
end
% m = 2;
n=ceil(M/m);

colormap(gray)

buf=1;
array=-ones(buf+m*(xsz+buf),buf+n*(ysz+buf));

k=1;
for i=1:m
    for j=1:n
        if k>M continue; end
        clim=max(abs(A(:,k)));
        array(buf+(i-1)*(xsz+buf)+[1:xsz],buf+(j-1)*(ysz+buf)+[1:ysz])=...
            reshape(A(:,k),xsz,ysz)/clim;
        k=k+1;
    end
end

if isreal(array)
    h=imagesc(array,'EraseMode','none',[-1 1]);
else
    h=imagesc(20*log10(abs(array)),'EraseMode','none',[-1 1]);
end;
axis image off

drawnow