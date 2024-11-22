import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
import numpy as np
import sys


class MatchingLoss(nn.Module):
    def __init__(self, loss_type='l1', is_weighted=False):
        super().__init__()
        self.is_weighted = is_weighted

        if loss_type == 'l1':
            self.loss_fn = F.l1_loss
        elif loss_type == 'l2':
            self.loss_fn = F.mse_loss
        else:
            raise ValueError(f'invalid loss type {loss_type}')

    def forward(self, predict, target, weights=None):

        loss = self.loss_fn(predict, target, reduction='none')
        loss = einops.reduce(loss, 'b ... -> b (...)', 'mean')

        if self.is_weighted and weights is not None:
            loss = weights * loss

        return loss.mean()
    

#----------------------------------------------------------------------------
                                # CDM (Martingale) Loss
#----------------------------------------------------------------------------


# @persistence.persistent_class
class MartingaleLoss:
    """ensures that the expected generated image is not changing."""
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5, epsilon_min=0.0, epsilon_max=0.05, num_steps=6, rho=7, 
        martingale_lambda=2., S_churn=10.0, S_min=0.01, S_max=1.0, S_noise=1.007):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.num_steps = num_steps
        self.rho = rho
        self.epsilon_min = epsilon_min
        self.epsilon_max = epsilon_max
        self.martingale_lambda = martingale_lambda
        
        self.S_churn = S_churn
        self.S_min = S_min
        self.S_max = S_max
        self.S_noise = S_noise

    def __call__(self, net, images, labels=None, augment_pipe=None):
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        # save_image(images, "before_augmentations.png")
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        # save_image(y, "after_augmentations.png")

        n = torch.randn_like(y) * sigma

        # Regular loss computation
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn_initial = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss1 = weight * ((D_yn_initial - y) ** 2)


        # repeat one time
        labels = labels.repeat([2, 1])
        augment_labels = augment_labels.repeat([2, 1])
        sigma_max = torch.clone(sigma)

        # get sigma deviations
        stretch = self.epsilon_max - self.epsilon_min
        epsilon = torch.rand_like(sigma) * stretch
        # TODO(@giannisdaras): fix this hardcoded value
        sigma_min = torch.maximum(sigma - epsilon, torch.ones_like(sigma) * 0.002)

        t_steps = edm_schedule(sigma_max=torch.unsqueeze(sigma_max, 1), sigma_min=torch.unsqueeze(sigma_min, 1), num_steps=self.num_steps)
        t_steps = t_steps.repeat([2, 1, 1, 1, 1])
        x_next = y.repeat([2, 1, 1, 1]) + n.repeat([2, 1, 1, 1])
        
        with torch.no_grad():
            for i in range(self.num_steps - 1):
                t_cur = t_steps[:, :, :, :, i]
                t_next = t_steps[:, :, :, :, i + 1]
                x_cur = x_next
                x_next, _ = backward_sde_sampler(net, x_cur, labels, self.num_steps, t_cur, t_next, i, second_order=False, augment_labels=augment_labels)
        
        D_yn = net(x_next, sigma_min.repeat([2, 1, 1, 1]), labels, augment_labels=augment_labels)
        x_hat_1 = D_yn[:D_yn.shape[0] // 2]
        x_hat_2 = D_yn[-D_yn.shape[0] // 2:]
        loss2 = (x_hat_1 - D_yn_initial) * (x_hat_2 - D_yn_initial)
        return loss1 + self.martingale_lambda * loss2


