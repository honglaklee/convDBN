%%% compute gradient w.r.t. weight tensor between visible and hidden units

function PAR = crbm_vishidprod(PAR, params, opt)
if ~exist('opt','var'), opt = 'pos'; end

selidx1 = size(PAR.hidprobs,1):-1:1;
selidx2 = size(PAR.hidprobs,2):-1:1;

if strcmp(opt,'pos'),
    %%% --- positive phase --- %%%
    for c = 1:params.numvis,
        for b = 1:params.numhid,
            PAR.posprods(:,:,c,b) = conv2(PAR.vis(:,:,c), PAR.hidprobs(selidx1, selidx2, b), 'valid');
        end
    end
elseif strcmp(opt,'neg'),
    %%% --- negative phase --- %%%
    for c = 1:params.numvis,
        for b = 1:params.numhid,
            PAR.negprods(:,:,c,b) = conv2(PAR.negdata(:,:,c), PAR.hidprobs(selidx1, selidx2, b), 'valid');
        end
    end
end

return