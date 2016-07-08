%%% hidden unit inference 
%%% of convolutional restricted Boltzmann machine
%%% with probabilistic max-pooling

function PAR = crbm_inference(CRBM, PAR, params, opt)
%
if ~exist('opt','var'), opt = 'pos'; end

PAR.hidprobs = CRBM.hbiasmat;
if strcmp(opt,'pos'),
    %%% --- positive phase --- %%%
    for c = 1:params.numvis,
        for b = 1:params.numhid,
            PAR.hidprobs(:,:,b) = PAR.hidprobs(:,:,b) + conv2(PAR.vis(:,:,c), CRBM.Wlr(:,:,b,c), 'valid');
        end
    end
elseif strcmp(opt,'neg'),
    %%% --- negative phase --- %%%
    for c = 1:params.numvis,
        for b = 1:params.numhid,
            PAR.hidprobs(:,:,b) = PAR.hidprobs(:,:,b) + conv2(PAR.negdata(:,:,c), CRBM.Wlr(:,:,b,c), 'valid');
        end
    end
end

PAR.hidprobs = 1/(params.sigma^2).*PAR.hidprobs;
[PAR.hidstates, PAR.hidprobs] = sample_multrand(PAR.hidprobs, params);

return;
