"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
Adapted by: Gonçalo Oliveira (goncalo.de.oliveira@tecnico.ulisboa.pt)
Date: 18 Mar, 2022
"""
from __future__ import print_function

import torch
import torch.nn as nn


class NLL:
    def __init__(self):
        pass
    def __call__(self, Y_prob, Y):
        return -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli
    def __repr__(self):
        return "negative log bernoulli"


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature = 0.07, base_temperature = 0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features, labels):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
        Note:
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))
        assert len(features.shape) == 2

        batch_size  = features.shape[0]
        labels      = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        mask        = torch.eq(labels, labels.T).float().to(device)

        # compute logits
        anchor_dot_contrast = torch.div(torch.matmul(features, features.T), self.temperature)
        # anchor_dot_contrast has the product of every feature vector with all 
        # other feature vectors (including itself)
        logits_max, _       = torch.max(anchor_dot_contrast, dim = 1, keepdim = True)
        logits              = anchor_dot_contrast - logits_max.detach() 
        # logits is the same as anchor_dot_contrast but the max of each line
        # is subtracted from each element in that line. This is done for numerical stability

        # mask-out self-contrast cases
        logits_mask = torch.scatter(torch.ones_like(mask), 
                                    1,
                                    torch.arange(batch_size).view(-1, 1).to(device),
                                    0)
        mask        = mask * logits_mask

        # compute log_prob
        exp_logits  = torch.exp(logits) * logits_mask
        log_prob    = logits - torch.log(exp_logits.sum(1, keepdim=True))
        # use of the rule log(a/b) = log(a) - log(b)
        # logits = log(exp(logits)) so no need to apply the exponent to logits

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(batch_size).mean()

        return loss


if __name__ == "__main__":
    import random
    BATCH_SIZE  = 40
    N_VIEWS     = 2
    N_FEATURES  = 128
    
    loss_fn     = SupConLoss()
    features    = torch.randn(BATCH_SIZE * N_VIEWS, N_FEATURES, requires_grad = True)/1000
    labels      = torch.tensor([int(random.random() > .5) for i in range(BATCH_SIZE)])
    labels      = labels.repeat(2)
    
    print( loss_fn(features, labels) )
