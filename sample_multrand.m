function [H HP Hc HPc] = sample_multrand(poshidexp, params, spacing)
if ~exist('spacing','var'),
    spacing = params.spacing;
end

% poshidexp is 3d array
poshidprobs_mult = zeros(spacing^2+1, size(poshidexp,1)*size(poshidexp,2)*size(poshidexp,3)/spacing^2);
poshidprobs_mult(end,:) = 0;

for c = 1:spacing,
    for r = 1:spacing,
        temp = poshidexp(r:spacing:end, c:spacing:end, :);
        poshidprobs_mult((c-1)*spacing+r,:) = temp(:);
    end
end

% substract from max exponent to make values numerically more stable
poshidprobs_mult = bsxfun(@minus, poshidprobs_mult, max(poshidprobs_mult,[],1));
poshidprobs_mult = exp(poshidprobs_mult);

[S1 P1] = multrand_col(poshidprobs_mult');
S = S1';
P = P1';
clear S1 P1

% convert back to original sized matrix
H = zeros(size(poshidexp));
HP = zeros(size(poshidexp));
for c = 1:spacing,
    for r = 1:spacing,
        H(r:spacing:end, c:spacing:end, :) = reshape(S((c-1)*spacing+r,:), [size(H,1)/spacing, size(H,2)/spacing, size(H,3)]);
        HP(r:spacing:end, c:spacing:end, :) = reshape(P((c-1)*spacing+r,:), [size(H,1)/spacing, size(H,2)/spacing, size(H,3)]);
    end
end

if nargout >2
    Sc = sum(S(1:end-1,:));
    Pc = sum(P(1:end-1,:));
    Hc = reshape(Sc, [size(poshidexp,1)/spacing,size(poshidexp,2)/spacing,size(poshidexp,3)]);
    HPc = reshape(Pc, [size(poshidexp,1)/spacing,size(poshidexp,2)/spacing,size(poshidexp,3)]);
end


return

function [S P] = multrand_col(P)
% P is 2-d matrix: 2nd dimension is # of choices

sumP = sum(P,2);
P = bsxfun(@rdivide,P,sumP);

cumP = cumsum(P,2);
unifrnd = rand(size(P,1),1);
temp = bsxfun(@gt,cumP,unifrnd);
Sindx = diff(temp,1,2);
S = zeros(size(P));
S(:,1) = 1-sum(Sindx,2);
S(:,2:end) = Sindx;

return;

