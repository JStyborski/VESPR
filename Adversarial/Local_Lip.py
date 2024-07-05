import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class Local_Lip_Loss:

    def __init__(self, topNorm=1, botNorm=np.inf, reduction='mean'):
        """
        Initialize norms and reduction methods
        :param topNorm: [float] - Norm to use on the numerator ||f(x) - f(xp)||
        :param botNorm: [float] - Norm to use on the denominator ||x - xp||
        :param reduction: [string] - Type of reduction to apply across all batch samples
        """
        self.topNorm = topNorm
        self.botNorm = botNorm
        self.reduction = reduction

        if self.topNorm not in [1, 2, np.inf, 'kl']:
            raise ValueError(f'Unsupported norm {self.topNorm}')
        if self.botNorm not in [1, 2, np.inf]:
            raise ValueError(f'Unsupported norm {self.botNorm}')

    def forward(self, x, xp, fx, fxp):
        """
        Calculate local Lipschitz value, ||f(x) - f(xp)||_p / ||x - xp||_q
        :param x: [Pytorch tensor] [b x d] - Nominal input tensor
        :param xp: [Pytorch tensor] [b x d] - Adversarial input tensor of x
        :param fx: [Pytorch tensor] [b x d2] - Encoding output of nominal input
        :param fxp: [Pytorch tensor] [b x d2] - Encoding output of adversarial input
        :return:
        """

        # Calculate difference between input samples
        # Convert all tensor dimensions after batchsize dimension into a vector - N,C,H,W to N,C*H*W
        bot = torch.flatten(x - xp, start_dim=1)

        # Use KL divergence to calculate the difference between model outputs, then calculate Lipschitz
        # PyTorch KLDivLoss calculates reduction(ytrue*log(ytrue/ypred)) where reduction is some method of aggregating the results (sum, mean)
        # yt*log(yt/yp) = yt*(log(yt)-log(yp)) --> PyTorch expects yp to be in logspace already, such that you input log(yp) and yt
        if self.topNorm == 'kl':
            criterion_kl = nn.KLDivLoss(reduction='none')
            top = criterion_kl(F.log_softmax(fxp, dim=1), F.softmax(fx, dim=1))
            lolip = torch.sum(top, dim=1) / torch.norm(bot + 1e-6, dim=1, p=self.botNorm)

        # Calculate Lipschitz constant using regular norms - the top just uses output logits (no softmax)
        else:
            top = torch.flatten(fx, start_dim=1) - torch.flatten(fxp, start_dim=1)
            lolip = torch.norm(top, dim=1, p=self.topNorm) / torch.norm(bot + 1e-6, dim=1, p=self.botNorm)

        if self.reduction == 'mean':
            return torch.mean(lolip)
        elif self.reduction == 'sum':
            return torch.sum(lolip)
        elif self.reduction == 'none':
            return lolip


def maximize_local_lip(model, X, alpha=0.6/255, eps=4./255, topNorm=1, botNorm=np.inf, nRestarts=10, nSteps=10, batchSize=64,
                       outIdx=None, normOuts=False, randInit=True, noiseMag=None, xMin=None, xMax=None):
    """
    Iteratively search for input that maximizes local Lipschitz within a specified radius
    This function is similar to the FGSM_PGD script in the same folder, but modified to accommodate Lipschitz as loss
    This function assumes that model and X are on same device
    :param model: [Pytorch Model] - Callable object that takes an input and returns an output encoding
    :param X: [Pytorch tensor] [b x d] - Nominal input tensor
    :param alpha: [float] - Adversarial sample step size
    :param eps: [float] - Norm constraint bound for infinity norm of adversarial sample
    :param topNorm: [float] - Norm type to apply to ||f(x) - f(xp)||
    :param botNorm: [float] - Norm type to apply to ||x - xp||
    :param batchSize: [int] - Number of samples to load from the dataloader each iteration
    :param nRestarts: [int]- Number of PGD restarts to try
    :param nSteps: [int] - Number of PGD steps to run for maximizing local Lipschitz
    :param outIdx: [int] - Index corresponding to the desired output (set as None for only 1 output)
    :param normOuts: [Bool] - Boolean to normalize outputs (xOut and xAdvOut)
    :param randInit: [Bool] - Whether to start adversarial search with random offset
    :param noiseMag: [float] - Maximum random initial perturbation magnitude
    :param xMin: [float] - Minimum value of x_adv (useful for ensuring images are useable)
    :param xMax: [float] - Maximum value of x_adv (useful for ensuring images are useable)
    :return avgLolip: [float] - Average final local Lipschitz values across input samples
    :return advTens: [Pytorch tensor] [b x d] - Final adversarial samples tensor
    """

    # Instantiate Local_Lip_Loss
    lll = Local_Lip_Loss(topNorm, botNorm, 'none')
    
    # Make a loader using the X tensor input
    loader = torch.utils.data.DataLoader(X, batch_size=batchSize, shuffle=False)

    # Store model training state and ensure model doesn't update batchnorm or anything
    trainModel = model.training
    model.eval()
    model.zero_grad()

    for iter1, x in enumerate(loader):

        # Get the outputs of the model for x and detach (don't need gradients for x or xOut)
        xOut = model(x) if outIdx is None else model(x)[outIdx]
        if normOuts:
            xOut = xOut / torch.norm(xOut, dim=1, p=2, keepdim=True)
        xOut = xOut.detach()

        for iter2 in range(nRestarts):

            # Apply random initial perturbation to input (or don't)
            # Madry 2017 apply random perturbations over many runs to see if adv results were very different - they were not
            if randInit:
                if noiseMag is None:
                    noiseMag = eps
                noise = torch.zeros_like(x).uniform_(-noiseMag, noiseMag)
                # Clip noise to ensure it does not violate x_adv norm constraint and then apply to x
                noise = torch.clamp(noise, -eps, eps)
                xAdv = x + noise.to(x.device)
            else:
                xAdv = x

            # Ensure x_adv elements within appropriate bounds
            if xMin is not None or xMax is not None:
                xAdv = torch.clamp(xAdv, xMin, xMax)

            for _ in range(nSteps):

                # Reset the gradients of the adversarial input and model
                xAdv = xAdv.detach().requires_grad_(True)

                # Get outputs of the model for x_adv, calculate the local lipschitz constant using x and xAdv, then backpropagate
                xAdvOut = model(xAdv) if outIdx is None else model(xAdv)[outIdx]
                if normOuts:
                    xAdvOut = xAdvOut / torch.norm(xAdvOut, dim=1, p=2, keepdim=True)
                lolip = lll.forward(x, xAdv, xOut, xAdvOut)
                sumlolip = torch.sum(lolip)
                model.zero_grad()
                sumlolip.backward()

                # Calculate the new adversarial example given the new step - gradient ascent towards higher Lipschitz
                # x_adv is detached, since x_adv.data and x_adv.grad do not carry autograd
                xAdv = xAdv.data + alpha * xAdv.grad.sign()

                # Clip total perturbation (measured from center x) to norm ball associated with x_adv limit
                eta = torch.clamp(xAdv - x, -eps, eps)
                xAdv = x + eta

                # Ensure x_adv elements within appropriate bounds
                if xMin is not None or xMax is not None:
                    xAdv = torch.clamp(xAdv, xMin, xMax)

            # Compare max lolip so far for each sample with lolip for this restart and take max
            if iter2 == 0:
                bestLolip = lolip.detach()
                xAdvFinal = xAdv.detach()
            else:
                betterLolipMask = lolip > bestLolip
                bestLolip[betterLolipMask] = lolip[betterLolipMask].detach()
                xAdvFinal[betterLolipMask, :] = xAdv[betterLolipMask, :].detach()

        if iter1 == 0:
            totalLolip = torch.sum(bestLolip).item()
            cumulAdvTens = xAdvFinal.detach()
        else:
            totalLolip += torch.sum(bestLolip).item()
            cumulAdvTens = torch.cat((cumulAdvTens, xAdvFinal.detach()), dim=0)

    # Return model to initial state
    model.train() if trainModel else model.eval()
    model.zero_grad()

    return totalLolip / len(X), cumulAdvTens