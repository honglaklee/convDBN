%%% visible unit inference (reconstruction)
%%% of convolutional restricted Boltzmann machine
%%% with probabilistic max-pooling

function PAR = crbm_reconstruct(CRBM, PAR, params, opt)

if ~exist('opt','var'), opt = 'neg'; end

if strcmp(opt,'recon'),
    %%% --- reconstruction --- %%%
    PAR.reconst = CRBM.vbiasmat;
    for b = 1:params.numhid,
        for c = 1:params.numvis,
            PAR.reconst(:,:,c) = PAR.reconst(:,:,c) + conv2(PAR.hidprobs(:,:,b), CRBM.W(:,:,c,b), 'full');
        end
    end
    
    if strcmp(params.intype,'binary'),
        PAR.reconst = (1/params.sigma^2)*PAR.reconst;
        PAR.reconst = sigmoid(PAR.reconst);
    end
elseif strcmp(opt,'neg'),
    %%% --- negative phase --- %%%
    PAR.negdata = CRBM.vbiasmat;
    for b = 1:params.numhid,
        for c = 1:params.numvis,
            PAR.negdata(:,:,c) = PAR.negdata(:,:,c) + conv2(PAR.hidstates(:,:,b), CRBM.W(:,:,c,b), 'full');
        end
    end
    
    if strcmp(params.intype,'binary'),
        PAR.negdata = (1/params.sigma^2)*PAR.negdata;
        PAR.negdata = sigmoid(PAR.negdata);
    end
end

return
