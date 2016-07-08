function [CRBM, PAR] = fobj_sparsity(CRBM, PAR, params, opt)

% sparsity regularizer
if isempty(PAR.runningavg_prob),
    PAR.runningavg_prob = PAR.poshidact/opt.hidsize;
else
    PAR.runningavg_prob = params.eta_sparsity*PAR.runningavg_prob + (1-params.eta_sparsity)*PAR.poshidact/opt.hidsize;
end

dh_reg = params.pbias - PAR.runningavg_prob;

% update
PAR.dh_total = PAR.dh_total + params.plambda*dh_reg;

return;
