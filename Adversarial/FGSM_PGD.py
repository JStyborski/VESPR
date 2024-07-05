"""The Projected Gradient Descent attack."""

import numpy as np
import torch

import Utils.Misc_Functions as MF

def scale_tens(tens, tens_idx, norm, eps):
    # If ||tens||_p > eps, scales all tens values such that ||tens||_p = eps
    # This function follows the torch.nn.utils.clip_grad implementation of norm scaling
    # :tens (tensor): the input tensor to modify based on its norm
    # :tens_idx (list): the list of tens dimension indices along which to calculate norms - dimensions not included will have a separate norm value for each element
    # :norm (float, int, or 'inf'): the type of norm to apply to tens
    # :eps (float or int): the maximum allowable norm value of tens
    # :returns: scaled_tens

    if tens_idx is None:
        tens_idx = list(range(1, len(tens.size())))

    # Get the norm of tens across the specified indices
    tens_norm = torch.norm(tens, dim=tens_idx, keepdim=True, p=norm)
    
    # If eps > eta_norm in certain elements, don't upscale them
    scale_coef = torch.clamp(eps / (tens_norm + 1e-6), max=1.0)
    scaled_tens = tens * scale_coef
    
    return scaled_tens

def clip_tens(tens, tens_idx, norm, eps):
    # If ||tens||_inf > eps, projects tens tensor to the nearest point where ||tens||_inf = eps
    # :tens (Tensor): the input tensor to modify based on its norm
    # :tens_idx (list): the list of tens dimension indices along which to calculate norms - dimensions not included will have a separate norm value for each element
    # :norm (float, int, or 'inf'): the type of norm to apply to tens
    # :eps (float): the maximum allowable inf norm value of tens
    # :returns clipped_tens (Tensor): 
    
    # Inf norm of tens cannot exceed eps - corresponds to clipping all values of tens beyond +/- eps
    if norm == np.inf:
        clipped_tens = torch.clamp(tens, min=-eps, max=eps)
    
    # Scaling isn't the same idea as clipping, but in practice, people will refer to scaling as clipping
    # I keep this part to ensure the function still works if people want to "clip" for other norms
    # Note that in 2-norm, clipping and scaling are identical processes
    else:

        if tens_idx is None:
            tens_idx = list(range(1, len(tens.size())))
        clipped_tens = scale_tens(tens, tens_idx, norm, eps)
    
    return clipped_tens

def optimize_linear(grad, grad_idx=None, norm=np.inf):
    # Solves for the optimal input to a linear function under a norm constraint.
    # Optimal_perturbation = argmax_eta,||eta||_p<=1(dot(eta, grad))
    # i.e., Find eta s.t. pth norm of eta <= 1 such that dot(eta, grad) is maximized
    # :grad (Tensor): batch of gradients
    # :grad_idx (list): the list of grad dimension indices along which to calculate norms - dimensions not included will have a separate norm value for each element
    # :norm (number): np.inf, 1, or 2. Order of norm constraint.
    # :returns eta (Tensor): optimal perturbation, the eta where ||eta||_p <= 1

    # dot(eta, grad) with ||eta||inf = max(abs(eta)) <= 1 is maximized when eta=sign(grad)
    # Optimal inf-norm constrained perturbation direction is the max magnitude grad value in every dimension
    if norm == np.inf:
        eta = torch.sign(grad)
    
    # dot(eta, grad) with ||eta||1 = sum(abs(eta_i)) <= 1 is maximized when eta is a +/- 1-hot corresponding to the maximum magnitude value of grad
    # Optimal 1-norm constrained perturbation direction is the max magnitude pixel value in any dimension
    elif norm == 1:

        if grad_idx is None:
            grad_idx = list(range(1, len(grad.size())))
    
        # Absolute value and sign tensors of gradient, used later
        abs_grad = torch.abs(grad)

        # Get the maximum values of the tensor as well as their locations
        # Max is executed across all dimensions except the first (batch dim), such that one max value is found for each batch sample
        max_abs_grad = torch.amax(abs_grad, dim=grad_idx, keepdim=True)
        max_mask = abs_grad.eq(max_abs_grad).to(torch.float)
        
        # Count the number of tied values at maximum for each batch sample
        num_ties = torch.sum(max_mask, dim=grad_idx, keepdim=True)
        
        # Optimal perturbation for 1-norm is only along the maximum magnitude dimensions
        # Hack to apply smaller perturbation in >1 dimensions if num_ties > 1
        eta = torch.sign(grad) * max_mask / num_ties
    
    # dot(eta, grad) with ||eta||2 = sqrt(sum(eta_i^2)) <= 1 is maximized when eta is equal to the normalized gradient vector
    # Optimal 2-norm constrained perturbation direction is Euclidean scaling along the gradient vector
    elif norm == 2:

        if grad_idx is None:
            grad_idx = list(range(1, len(grad.size())))
    
        # Get the 2 norm of the gradient vector for each batch sample and normalize
        eta = grad / torch.norm(grad, dim=grad_idx, keepdim=True, p=2)
        
    else:
        raise NotImplementedError('Only L-inf, L1 and L2 norms are currently implemented.')

    return eta


def sl_pgd(model, lossFn, useAMP, X, Y, alpha, eps, norm, nRestarts, nSteps, batchSize, outIdx=None,
           targeted=False, randInit=True, noiseMag=None, xMin=None, xMax=None):
    """
    Implementation the Kurakin 2016 Basic Iterative Method (rand_init=False) or Madry 2017 PGD method (rand_init=True)
    This function assumes that model and x are on same device
    Equivalent to FGSM if eps is high such that no limit is applied, rand_init = False, and n_restarts = perturb_steps = 1
    :param model: [function] [1] - Callable function that takes an input tensor and returns the model logits
    :param lossFn: [function] [1] - Callable function for calculating loss values per input - the loss function should be initialized already with
        reduction='none' so that each input gets a loss
        - CrossEntropyLoss - Typical loss used for FGSM/PGD. The main issue is when logit magnitudes are large (e.g., 500) such that softmax(logits)
        exponential is dominated by 1 term and the output distribution is a one-hot. If the one-hot is correct, then CE loss is 0, then the gradient
        is 0, then eta is 0, then x_adv = x. Typical remedies are to L2 normalize logits or use NLLLoss (below)
        - NLLLoss - Not typically used for FGSM/PGD. This essentially returns the negative of the correct logit as loss.
    :param useAMP: [Bool] - Boolean to use mixed precision on forward prop or not
    :param X: [Pytorch tensor] [m x n] - Nominal input tensor
    :param Y: [tensor] [m] - Tensor with truth/target labels
    :param alpha: [float] [1] - Input variation parameter, see https://arxiv.org/abs/1412.6572
    :param eps: [float] [1] - Norm constraint bound for adversarial example
    :param norm: [float] [1] - Order of the norm (mimics NumPy)
    :param nRestarts: [int] [1] - Number of PGD restarts
    :param nSteps: [int] [1] - Number of PGD steps
    :param batchSize: [int] [1] - Number of samples to calculate loss for at once
    :param outIdx: [int] [1] - Index corresponding to the desired output (set as None for only 1 output)
    :param targeted: [Bool] [1] - Whether to direct the adversarial attack towards a specific label/target
    :param randInit: [Bool] [1] - Whether to start adversarial search with random offset
    :param noiseMag: [float] [1] - Maximum random initial perturbation magnitude
    :param xMin: [float] [1] - Minimum value of x_adv (useful for ensuring images are useable)
    :param xMax: [float] [1] - Maximum value of x_adv (useful for ensuring images are useable)
    :return: Adversarial tensor
    """

    if norm not in [np.inf, 1, 2]:
        raise ValueError(f'Unsupported norm {norm}')
    if norm == 1:
        print('Warning: FGM may not be a good inner loop step, because norm=1 FGM only changes 1 pixel at a time')

    # Make a dataset and loader using the X and Y tensor inputs
    dataset = torch.utils.data.TensorDataset(X, Y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batchSize, shuffle=False)

    # Store model training state and ensure model doesn't update batchnorm or anything
    trainModel = model.training
    model.eval()
    model.zero_grad()

    for iter1, (x, y) in enumerate(loader):

        for iter2 in range(nRestarts):

            # Apply random initial perturbation to input (or don't)
            # Madry 2017 apply random perturbations over many runs to see if adv results were very different - they were not
            if randInit:
                if noiseMag is None:
                    noiseMag = eps
                noise = torch.zeros_like(x).uniform_(-noiseMag, noiseMag)
                # Clip noise to ensure it does not violate x_adv norm constraint and then apply to x
                noise = clip_tens(noise, None, norm, eps)
                xAdv = x + noise.to(x.device)
            else:
                xAdv = x

            # Ensure x_adv elements within appropriate bounds
            if xMin is not None or xMax is not None:
                xAdv = torch.clamp(xAdv, xMin, xMax)

            for _ in range(nSteps):

                # Reset the gradients of the adversarial input
                xAdv = xAdv.detach().requires_grad_(True)

                with torch.cuda.amp.autocast(enabled=useAMP):

                    # Calculate loss and get loss gradient
                    # If attack is targeted, define loss such that gradient dL/dx will point towards target class
                    # else, define loss such that gradient dL/dx will point away from correct class
                    lossVal = lossFn(model(xAdv), y) if outIdx is None else lossFn(model(xAdv)[outIdx], y)
                    if targeted:
                        lossVal = -lossVal
                    sumLoss = torch.sum(lossVal)

                model.zero_grad()
                sumLoss.backward()

                # In tests on CIFAR-10 and ImageNet-100, no loss scaling worked just as well as my heuristic scaling fix below
                #lossScaler.scale(sumLoss).backward()  # Backprop on upscaled loss to prevent underflow with AMP
                # Input gradients are not included in optimizer so gradient unscaling is done manually and lossScaler is not stepped/updated
                #unscaledGrads = xAdv.grad * lossScaler._scale.double().reciprocal().float()
                # If lossScaler does not detect NaNs/infs in optimizer for _growth_interval batches, it will increase _scale
                # This sometimes causes NaNs/infs in the input gradients but not in the optimizer
                # Because input gradients are not included in optimizer, lossScaler protocols do not check input gradients for NaNs/infs
                # Below heuristic checks for NaNs/infs in input gradients and if any are found, reduces _scale manually
                # This does not reset progress along _growth_interval or anything else, only changes _scale
                # If a NaN/inf occurs in unscaledGrads, the corresponding eta value is 0., (i.e., no adv perturbation on that element)
                # This seems like a harmless consequence for only one adversarial step, so I permit it
                #if lossScaler._scale > 1.0 and not torch.all(torch.isfinite(unscaledGrads)):
                #    lossScaler.update(new_scale=lossScaler._scale * lossScaler._backoff_factor)

                # eta is the norm-constrained direction that maximizes dot(perturbation, x.grad)
                # eta is detached, since x_adv.grad does not carry autograd
                eta = optimize_linear(xAdv.grad, None, norm)

                # Add perturbation to original example to step away from correct label and obtain adversarial example
                # x_adv is detached, since x_adv.data does not carry autograd
                xAdv = xAdv.data + alpha * eta

                # Clip total perturbation (measured from center x) to norm ball associated with x_adv limit
                eta = clip_tens(xAdv - x, None, norm, eps)
                xAdv = x + eta

                # Ensure x_adv elements within appropriate bounds
                if xMin is not None or xMax is not None:
                    xAdv = torch.clamp(xAdv, xMin, xMax)

            # Compare the best loss so far to loss for this restart and take the better adversarial sample
            if iter2 == 0:
                bestLoss = lossVal.detach()
                xAdvFinal = xAdv.detach()
            else:
                if targeted:
                    betterLossMask = lossVal.detach() < bestLoss
                else:
                    betterLossMask = lossVal.detach() > bestLoss
                bestLoss[betterLossMask] = lossVal[betterLossMask].detach()
                xAdvFinal[betterLossMask, :] = xAdv[betterLossMask, :].detach()

        # Calculate loss for each sample and sum batch and append the adversarial tensor to list for concat later
        if iter1 == 0:
            totalLoss = torch.sum(bestLoss).item()
            cumulAdvTens = xAdvFinal.detach()
        else:
            totalLoss += torch.sum(bestLoss).item()
            cumulAdvTens = torch.cat((cumulAdvTens, xAdvFinal.detach()), dim=0)

    # Return model to initial state
    model.train() if trainModel else model.eval()
    model.zero_grad()

    return totalLoss / len(X), cumulAdvTens


def ssl_pgd(model, lossFn, useAMP, inpList, useAdvList, gatherTensors, alpha, eps, norm, nRestarts, nSteps, batchSize,
            targeted=False, randInit=True, noiseMag=None, xMin=None, xMax=None):
    """
    Implementation the Kurakin 2016 Basic Iterative Method (rand_init=False) or Madry 2017 PGD method (rand_init=True)
    This function assumes that model and x are on same device
    Equivalent to FGSM if eps is high such that no limit is applied, rand_init = False, and n_restarts = perturb_steps = 1
    :param model: [function] [1] - Callable function that takes an input tensor and returns the model logits
    :param lossFn: [function] [1] - Callable function for calculating SSL loss values from an output list.
        Contrary to sl_pgd, this function expects a single loss value output (i.e., reduction is 'mean' or 'sum'). This is because sl_pgd samples are
        evaluated independently against a truth label, whereas ssl_pgd samples are evaluated against each other (e.g., InfoNCE). It could be incorrect
        to update only some samples. Also, tracking individual losses across multiple views is difficult.
    :param useAMP: [Bool] - Boolean to use mixed precision on forward prop or not
    :param inpList: [list] [n] - List of augmentation tensors, each a view used in the SSL input
    :param useAdvList: [list] [n] - List of booleans for each view, whether to apply adversarial
    :param gatherTensors: [Bool] [1] - Whether to gather tensors from across GPUs for loss calculation
    :param alpha: [float] [1] - Input variation parameter, see https://arxiv.org/abs/1412.6572
    :param eps: [float] [1] - Norm constraint bound for adversarial example
    :param norm: [float] [1] - Order of the norm (mimics NumPy)
    :param nRestarts: [int] [1] - Number of PGD restarts
    :param nSteps: [int] [1] - Number of PGD steps
    :param batchSize: [int] [1] - Number of samples to calculate loss for at once
    :param targeted: [Bool] [1] - Whether to direct the adversarial attack towards a specific label/target
    :param randInit: [Bool] [1] - Whether to start adversarial search with random offset
    :param noiseMag: [float] [1] - Maximum random initial perturbation magnitude
    :param xMin: [float] [1] - Minimum value of x_adv (useful for ensuring images are useable)
    :param xMax: [float] [1] - Maximum value of x_adv (useful for ensuring images are useable)
    :return: Adversarial tensors
    """

    if norm not in [np.inf, 1, 2]:
        raise ValueError(f'Unsupported norm {norm}')
    if norm == 1:
        print('Warning: FGM may not be a good inner loop step, because norm=1 FGM only changes 1 pixel at a time')

    # Make a dataset and loader using the X and Y tensor inputs
    dataset = torch.utils.data.TensorDataset(*inpList)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batchSize, shuffle=False)

    # Store model training state and ensure model doesn't update batchnorm or anything
    trainModel = model.training
    model.eval()
    model.zero_grad()

    for iter1, augList in enumerate(loader):

        for iter2 in range(nRestarts):

            # Loop through the augmentation tensors to initialize adversarial tensors
            advList = []
            for i, augTens in enumerate(augList):

                # Apply random initial perturbation to input (or don't)
                if randInit and useAdvList[i]:
                    if noiseMag is None:
                        noiseMag = eps
                    noise = torch.zeros_like(augTens).uniform_(-noiseMag, noiseMag)
                    noise = clip_tens(noise, None, norm, eps)
                    advTens = augTens + noise.to(augTens.device)
                else:
                    advTens = augTens

                # Ensure adv elements within appropriate bounds
                if xMin is not None or xMax is not None:
                    advTens = torch.clamp(advTens, xMin, xMax)

                advList.append(advTens)

            for _ in range(nSteps):

                # Reset the gradients of the adversarial inputs
                for i in range(len(advList)):
                    advList[i] = advList[i].detach()
                    if useAdvList[i]:
                        advList[i].requires_grad = True

                with torch.cuda.amp.autocast(enabled=useAMP):

                    outList = []
                    for i in range(len(advList)):

                        # Calculate outputs and append
                        pAdv, _, _, mzAdv = model(advList[i])
                        if gatherTensors:
                            pAdv = torch.cat(MF.FullGatherLayer.apply(pAdv.contiguous()), dim=0)
                            mzAdv = torch.cat(MF.FullGatherLayer.apply(mzAdv.contiguous()), dim=0)
                        outList.append([pAdv, mzAdv])

                    # Calculate loss
                    lossVal = lossFn(outList)
                    if targeted:
                        lossVal = -lossVal

                # Backpropagation
                model.zero_grad()
                lossVal.backward()  # No loss scaling despite multi-precision, following results on CIFAR-10 and IN-100 (see comments in sl_pgd)

                # Loop through adversarial tensors
                for i in range(len(advList)):

                    # Update adversarial tensors based on backpropagated gradients
                    if useAdvList[i]:
                        eta = optimize_linear(advList[i].grad, None, norm)
                        advList[i] = advList[i].data + alpha * eta
                        eta = clip_tens(advList[i] - augList[i], None, norm, eps)
                        advList[i] = augList[i] + eta

                    # Ensure adv elements within appropriate bounds
                    if xMin is not None or xMax is not None:
                        advList[i] = torch.clamp(advList[i], xMin, xMax)

            # Compare the best loss so far to loss for this restart and take the better adversarial sample
            if iter2 == 0:
                bestLoss = lossVal.detach()
                finalAdvList = [adv.detach() for adv in advList]
            else:
                # If new loss is better, update best loss and finalAdvList
                if (targeted and lossVal.detach() < bestLoss) or (not targeted and lossVal.detach() > bestLoss):
                    bestLoss = lossVal.detach()
                    finalAdvList = [adv.detach() for adv in advList]

        # Calculate loss for each sample and sum batch and append the adversarial tensor to list for concat later
        if iter1 == 0:
            totalLoss = bestLoss.item()
            cumulAdvList = finalAdvList
        else:
            totalLoss += bestLoss.item()
            for i in range(len(cumulAdvList)):
                cumulAdvList[i] = torch.cat((cumulAdvList[i], finalAdvList[i]), dim=0)

    # Return model to initial state
    model.train() if trainModel else model.eval()
    model.zero_grad()

    return totalLoss / (iter1 + 1), cumulAdvList


def ssl_sl_pgd(model, lossFn1, lossFn2, lossWtAlpha, lossWtBeta, useAMP, inpList, labelTens, useAdvList, gatherTensors, alpha, eps, norm,
               nRestarts, nSteps, batchSize, targeted=False, randInit=True, noiseMag=None, xMin=None, xMax=None):
    """
    Implementation the Kurakin 2016 Basic Iterative Method (rand_init=False) or Madry 2017 PGD method (rand_init=True)
    This function assumes that model and x are on same device
    Equivalent to FGSM if eps is high such that no limit is applied, rand_init = False, and n_restarts = perturb_steps = 1
    :param model: [function] [1] - Callable function that takes an input tensor and returns the model logits
    :param lossFn1: [function] [1] - Callable function for calculating SSL loss values from an output list.
            Contrary to sl_pgd, this function expects a single loss value output (i.e., reduction is 'mean' or 'sum')
    :param lossFn2: [function] [1] - Callable function for calculating SSL loss values from an output list.
            Contrary to sl_pgd, this function expects a single loss value output (i.e., reduction is 'mean' or 'sum')
    :param lossWtAlpha: [function] - Weighting coefficient for lossFn1
    :param lossWtBeta: [function] - Weighting coefficient for lossFn2
    :param useAMP: [Bool] - Boolean to use mixed precision on forward prop or not
    :param inpList: [list] [n] - List of augmentation tensors, each a view used in the SSL input
    :param useAdvList: [list] [n] - List of booleans for each view, whether to apply adversarial
    :param gatherTensors: [Bool] [1] - Whether to gather tensors from across GPUs for loss calculation
    :param alpha: [float] [1] - Input variation parameter, see https://arxiv.org/abs/1412.6572
    :param eps: [float] [1] - Norm constraint bound for adversarial example
    :param norm: [float] [1] - Order of the norm (mimics NumPy)
    :param nRestarts: [int] [1] - Number of PGD restarts
    :param nSteps: [int] [1] - Number of PGD steps
    :param batchSize: [int] [1] - Number of samples to calculate loss for at once
    :param targeted: [Bool] [1] - Whether to direct the adversarial attack towards a specific label/target
    :param randInit: [Bool] [1] - Whether to start adversarial search with random offset
    :param noiseMag: [float] [1] - Maximum random initial perturbation magnitude
    :param xMin: [float] [1] - Minimum value of x_adv (useful for ensuring images are useable)
    :param xMax: [float] [1] - Maximum value of x_adv (useful for ensuring images are useable)
    :return: Adversarial tensors
    """

    if norm not in [np.inf, 1, 2]:
        raise ValueError(f'Unsupported norm {norm}')
    if norm == 1:
        print('Warning: FGM may not be a good inner loop step, because norm=1 FGM only changes 1 pixel at a time')

    # Make a dataset and loader using the X and Y tensor inputs
    dataset = torch.utils.data.TensorDataset(*inpList, labelTens)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batchSize, shuffle=False)

    # Store model training state and ensure model doesn't update batchnorm or anything
    trainModel = model.training
    model.eval()
    model.zero_grad()

    for iter1, iterData in enumerate(loader):

        augList = iterData[:-1]
        truthTens = iterData[-1]

        # Gather targets to match gathered predictions
        if gatherTensors:
            truthTens = torch.cat(MF.FullGatherLayer.apply(truthTens.contiguous()), dim=0)

        for iter2 in range(nRestarts):

            # Loop through the augmentation tensors to initialize adversarial tensors
            advList = []
            for i, augTens in enumerate(augList):

                # Apply random initial perturbation to input (or don't)
                if randInit and useAdvList[i]:
                    if noiseMag is None:
                        noiseMag = eps
                    noise = torch.zeros_like(augTens).uniform_(-noiseMag, noiseMag)
                    noise = clip_tens(noise, None, norm, eps)
                    advTens = augTens + noise.to(augTens.device)
                else:
                    advTens = augTens

                # Ensure adv elements within appropriate bounds
                if xMin is not None or xMax is not None:
                    advTens = torch.clamp(advTens, xMin, xMax)

                advList.append(advTens)

            for _ in range(nSteps):

                # Reset the gradients of the adversarial inputs
                for i in range(len(advList)):
                    advList[i] = advList[i].detach()
                    if useAdvList[i]:
                        advList[i].requires_grad = True

                with torch.cuda.amp.autocast(enabled=useAMP):

                    outList1 = []
                    outList2 = []
                    for i in range(len(advList)):

                        # Calculate outputs and append
                        pAdv, _, _, mzAdv, cAdv = model(advList[i])
                        if gatherTensors:
                            pAdv = torch.cat(MF.FullGatherLayer.apply(pAdv.contiguous()), dim=0)
                            mzAdv = torch.cat(MF.FullGatherLayer.apply(mzAdv.contiguous()), dim=0)
                            cAdv = torch.cat(MF.FullGatherLayer.apply(cAdv.contiguous()), dim=0)
                        outList1.append([pAdv, mzAdv])
                        outList2.append(cAdv)

                    # Calculate loss and backpropagate
                    lossVal1 = lossFn1(outList1)
                    lossVal2, _, _ = lossFn2(outList2, truthTens)
                    lossVal = lossWtAlpha * lossVal1 + lossWtBeta * lossVal2
                    if targeted:
                        lossVal = -lossVal

                model.zero_grad()
                lossVal.backward()  # No loss scaling despite multi-precision, following results on CIFAR-10 and IN-100 (see comments in sl_pgd)

                # Loop through adversarial tensors
                for i in range(len(advList)):

                    # Update adversarial tensors based on backpropagated gradients
                    if useAdvList[i]:
                        eta = optimize_linear(advList[i].grad, None, norm)
                        advList[i] = advList[i].data + alpha * eta
                        eta = clip_tens(advList[i] - augList[i], None, norm, eps)
                        advList[i] = augList[i] + eta

                    # Ensure adv elements within appropriate bounds
                    if xMin is not None or xMax is not None:
                        advList[i] = torch.clamp(advList[i], xMin, xMax)

            # Compare the best loss so far to loss for this restart and take the better adversarial sample
            if iter2 == 0:
                bestLoss = lossVal.detach()
                finalAdvList = [adv.detach() for adv in advList]
            else:
                # If new loss is better, update best loss and finalAdvList
                if (targeted and lossVal.detach() < bestLoss) or (not targeted and lossVal.detach() > bestLoss):
                    bestLoss = lossVal.detach()
                    finalAdvList = [adv.detach() for adv in advList]

        # Calculate loss for each sample and sum batch and append the adversarial tensor to list for concat later
        if iter1 == 0:
            totalLoss = bestLoss.item()
            cumulAdvList = finalAdvList
        else:
            totalLoss += bestLoss.item()
            for i in range(len(cumulAdvList)):
                cumulAdvList[i] = torch.cat((cumulAdvList[i], finalAdvList[i]), dim=0)

    # Return model to initial state
    model.train() if trainModel else model.eval()
    model.zero_grad()

    return totalLoss / (iter1 + 1), cumulAdvList

