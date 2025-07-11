import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from onconet.models.pools.abstract_pool import AbstractPool
from onconet.models.pools.factory import RegisterPool
from onconet.models.pools.factory import get_pool
from onconet.utils.risk_factors import RiskFactorVectorizer

import pdb

MLP_HIDDEN_DIM = 100

@RegisterPool('RiskFactorPool')
class RiskFactorPool(AbstractPool):
    def __init__(self, args, num_chan):
        super(RiskFactorPool, self).__init__(args, num_chan)
        self.args = args
        self.internal_pool = get_pool(args.pool_name)(args, num_chan)
        
        assert not self.internal_pool.replaces_fc()
        self.dropout = nn.Dropout(args.dropout)
        self.length_risk_factor_vector = RiskFactorVectorizer(args).vector_length
        if self.args.pred_risk_factors and self.args.use_risk_factors:
            for key in args.risk_factor_keys:
                num_key_features = args.risk_factor_key_to_num_class[key]
                key_fc = nn.Linear(self.args.hidden_dim, num_key_features)
                self.add_module('{}_fc'.format(key), key_fc)

        self.args.img_only_dim = self.args.hidden_dim
        self.args.rf_dim = self.length_risk_factor_vector
        self.args.hidden_dim = self.args.rf_dim + self.args.img_only_dim

    def replaces_fc(self):
        return False

    def forward(self, x, risk_factors):

        if self.args.replace_snapshot_pool:
            x = x.data
        _, hidden = self.internal_pool(x)

        risk_factors_hidden = None
        # If valid risk_factors are provided
        if isinstance(risk_factors, (list, tuple)) and all(isinstance(rf, torch.Tensor) for rf in risk_factors):
            try:
                risk_factors_hidden = torch.cat(risk_factors, dim=1)
                print("[risk_factor_pool.py] Using provided risk factors of shape:", risk_factors_hidden.shape)
            except Exception as e:
                print("[risk_factor_pool.py] Failed to concatenate risk factors:", str(e))
                risk_factors_hidden = None
        else:
            print("[risk_factor_pool.py] No valid risk factors provided. Skipping RF usage.")
            # Fill with dummy zeros matching expected risk factor input (e.g., 100 dims)
            batch_size = hidden.shape[0]
            risk_vector_dim = 100  # <- set this based on your model's training config
            risk_factors_hidden = torch.zeros(batch_size, risk_vector_dim).to(hidden.device)
        # Only combine with hidden if risk_factors_hidden is available
        if risk_factors_hidden is not None:
            print("[risk_fcator_pool.py] The length of risk_factor_hidden is", len(risk_factors_hidden))
            print("[Before] hidden device:", hidden.device)
            print("[Before] risk_factors_hidden device:", risk_factors_hidden.device)

            # Align devices
            risk_factors_hidden = risk_factors_hidden.to(hidden.device)

            print("[After] hidden device:", hidden.device)
            print("[After] risk_factors_hidden device:", risk_factors_hidden.device)

            hidden = torch.cat((hidden, risk_factors_hidden), dim=1)
            print("[risk_fcator_pool.py] Risk factors combined with hidden:", hidden.shape)
        else:
            print("[risk_fcator_pool.py] Skipping concatenation as risk_factors_hidden is None.")

        hidden = self.dropout(hidden)
        print("[risk_fcator_pool.py] After dropout layer:", hidden.shape)
        return None, hidden
        #Below is the code for auto-inferring risk factors; above is the modified code to not
        # if self.args.pred_risk_factors:
        #     pred_risk_factors = []
        #     for indx, key in enumerate(self.args.risk_factor_keys):
        #         gold_rf = risk_factors[indx] if risk_factors is not None else None
        #         key_logit = self._modules['{}_fc'.format(key)](hidden)

        #         if self.args.risk_factor_key_to_num_class[key] == 1:
        #             key_probs = torch.sigmoid(key_logit)
        #         else:
        #             key_probs = F.softmax(key_logit, dim=-1)

        #         if not self.training and self.args.use_pred_risk_factors_if_unk:
        #             is_rf_known = (torch.sum(gold_rf, dim=-1) > 0).unsqueeze(-1).float()
        #             key_probs = (is_rf_known * gold_rf) + (1 - is_rf_known)*key_probs
        #         elif self.training and self.args.mask_prob > 0 and gold_rf is not None:
        #             is_rf_known = np.random.random() > self.args.mask_prob
        #             key_probs = (is_rf_known * gold_rf) + (1 - is_rf_known) * key_probs

        #         pred_risk_factors.append(key_probs)




        #     if (not self.training and self.args.use_pred_risk_factors_at_test) or (self.training and self.args.mask_prob > 0):
        #         risk_factors_hidden = torch.cat(pred_risk_factors, dim=1)

        # risk_factors_hidden = torch.cat(risk_factors, dim=1) if risk_factors_hidden is None else risk_factors_hidden
        # hidden = torch.cat((hidden, risk_factors_hidden), 1)
        # hidden = self.dropout(hidden)
        # return None, hidden


    def get_pred_rf_loss(self, hidden, risk_factors):
        #splice RF out from vector to get image based hidden
        img_hidden = hidden[:,:-self.length_risk_factor_vector]
        loss = 0
        num_losses = 0
        for i, key in enumerate(self.args.risk_factor_keys):
            key_logit = self._modules['{}_fc'.format(key)](img_hidden)
            key_gold = risk_factors[i]
            if self.args.risk_factor_key_to_num_class[key] == 1:
                loss += F.binary_cross_entropy_with_logits(key_logit, key_gold)
                num_losses += 1
            else:
                key_gold = key_gold.nonzero()
                if len(key_gold) == 0:
                    continue
                indicies_with_gold = key_gold[:,0].contiguous()
                key_logit = key_logit.index_select(dim=0, index=indicies_with_gold)
                key_gold = key_gold[:,-1:].contiguous().view(-1)
                loss += F.cross_entropy(key_logit, key_gold)
                num_losses += 1
        if num_losses > 0:
            loss /= num_losses
        return loss
