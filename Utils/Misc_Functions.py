import math
import torch

def collate_list(list_items):
    # Incoming list has shape batchSize x nOutputs, reshape to nOutputs x batchSize
    return [[sample[i] for sample in list_items] for i in range(len(list_items[0]))]

def gather_tensors(outLen, tens):
    outTens = torch.zeros(outLen, tens.size(1), device=tens.get_device())
    torch.distributed.all_gather_into_tensor(outTens, tens)
    return outTens

class FullGatherLayer(torch.autograd.Function):
    """
    Gather tensors from all process and support backward propagation for the gradients across processes.
    This function is taken directly from the official VICReg code: https://github.com/facebookresearch/vicreg/blob/main/main_vicreg.py
    I checked through the results already - using 4GPU with gather gives the same losses and gradients as using 1GPU with/without gather.
    async_op defaults to for all_gather and all_reduce, so no need to call barrier() or synchronize()
    """

    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        torch.distributed.all_reduce(all_gradients)
        return all_gradients[torch.distributed.get_rank()]


class LARC(object):
    """
    :class:`LARC` is a pytorch implementation of both the scaling and clipping variants of LARC,
    in which the ratio between gradient and parameter magnitudes is used to calculate an adaptive
    local learning rate for each individual parameter. The algorithm is designed to improve
    convergence of large batch training.

    See https://arxiv.org/abs/1708.03888 for calculation of the local learning rate.
    In practice it modifies the gradients of parameters as a proxy for modifying the learning rate
    of the parameters. This design allows it to be used as a wrapper around any torch.optim Optimizer.
    ```
    model = ...
    optim = torch.optim.Adam(model.parameters(), lr=...)
    optim = LARC(optim)
    ```
    It can even be used in conjunction with apex.fp16_utils.FP16_optimizer.
    ```
    model = ...
    optim = torch.optim.Adam(model.parameters(), lr=...)
    optim = LARC(optim)
    optim = apex.fp16_utils.FP16_Optimizer(optim)
    ```
    Args:
        optimizer: Pytorch optimizer to wrap and modify learning rate for.
        trust_coefficient: Trust coefficient for calculating the lr. See https://arxiv.org/abs/1708.03888
        clip: Decides between clipping or scaling mode of LARC. If `clip=True` the learning rate is set to `min(optimizer_lr, local_lr)` for each parameter. If `clip=False` the learning rate is set to `local_lr*optimizer_lr`.
        eps: epsilon kludge to help with numerical stability while calculating adaptive_lr
    """

    def __init__(self, optimizer, trust_coefficient=0.02, clip=True, eps=1e-8):
        self.optim = optimizer
        self.trust_coefficient = trust_coefficient
        self.eps = eps
        self.clip = clip

    def __getstate__(self):
        return self.optim.__getstate__()

    def __setstate__(self, state):
        self.optim.__setstate__(state)

    @property
    def state(self):
        return self.optim.state

    def __repr__(self):
        return self.optim.__repr__()

    @property
    def param_groups(self):
        return self.optim.param_groups

    @param_groups.setter
    def param_groups(self, value):
        self.optim.param_groups = value

    def state_dict(self):
        return self.optim.state_dict()

    def load_state_dict(self, state_dict):
        self.optim.load_state_dict(state_dict)

    def zero_grad(self):
        self.optim.zero_grad()

    def add_param_group(self, param_group):
        self.optim.add_param_group(param_group)

    def step(self):
        with torch.no_grad():
            weight_decays = []
            for group in self.optim.param_groups:
                # absorb weight decay control from optimizer
                weight_decay = group['weight_decay'] if 'weight_decay' in group else 0
                weight_decays.append(weight_decay)
                group['weight_decay'] = 0
                for p in group['params']:
                    if p.grad is None:
                        continue
                    param_norm = torch.norm(p.data)
                    grad_norm = torch.norm(p.grad.data)

                    if param_norm != 0 and grad_norm != 0:
                        # calculate adaptive lr + weight decay
                        adaptive_lr = self.trust_coefficient * (param_norm) / (grad_norm + param_norm * weight_decay + self.eps)

                        # clip learning rate for LARC
                        if self.clip:
                            # calculation of adaptive_lr so that when multiplied by lr it equals `min(adaptive_lr, lr)`
                            adaptive_lr = min(adaptive_lr / group['lr'], 1)

                        p.grad.data += weight_decay * p.data
                        p.grad.data *= adaptive_lr

        self.optim.step()
        # return weight decay control to optimizer
        for i, group in enumerate(self.optim.param_groups):
            group['weight_decay'] = weight_decays[i]

def spectral_filter(z, power=0.0, cutoff=None):
    # For z (n x d) and Cz (d x d), z = U Sigz V.T, Cz = 1/n z.T z = Q Lamz Q.T, with Q = V and Lamz = 1/n Sigz^2
    # Spectral filter g(Lam) adjusts eigenvalues and then applies W = V g(Lamz) V.T on z, p = z @ W
    # This affects output correlation: Cp = V g(Lamz)^2 Lam V.T, such that Lamp = g(Lamz)^2 Lamz
    # Low pass filter emphasizes large eigvals and diminishes low eigvals - high pass filter vice versa
    # In this function we specifically apply g(Lamz) = Lamz.pow(power)
    # power should be between -0.5 and +1.0 - [-0.5, 0] gives high pass filter, [0, 1.0] gives low pass filter
    # Power examples: -0.5 -> Lamp = I, 0 -> Lamp = Lamz, 0.5 -> Lamp = Lamz^2, 1.0 -> Lamp = Lamz^3
    U, Sigz, VT = torch.linalg.svd(z, full_matrices=False)
    Lamz = 1 / z.size(0) * Sigz.clamp(0).pow(2)
    Lamp = Lamz
    if power is not None:
        Lamp = Lamz.pow(1 + 2 * power)
    if cutoff is not None:
        Lamp[cutoff:] = 0
    Sigp = Lamp.sqrt() * z.size(0) ** 0.5
    specZ = U @ torch.diag(Sigp) @ VT
    return specZ

def cosine_evolve(epoch, startEpoch, nEpochs, startVal, finalVal, shiftFwd=False):
    # Evolve a value along a cosine curve from startVal to finalVal across nEpochs
    # This formula returns epoch=startEpoch returns currVal<=startVal and epoch=startEpoch+nEpochs returns currVal=finalVal
    # Unfortunately, this implies that after nEpochs of evolution, the value has not yet reached finalVal
    # e.g., startEpoch=0, nEpochs=10: currVal=/=finalVal when epoch=9 (the 10th epoch), currVal=finalVal when epoch=10 (the 11th epoch)
    # This convention is normal in other SSL codes (e.g., SimSiam)
    # shiftFwd shifts the evolution trend by 1 epoch, such that the value reaches finalVal 1 epoch earlier

    if shiftFwd:
        epoch += 1

    if epoch <= startEpoch:
        return startVal
    elif epoch >= startEpoch + nEpochs:
        return finalVal
    else:
        return finalVal + 0.5 * (startVal - finalVal) * (1. + math.cos(math.pi * (epoch - startEpoch) / nEpochs))

def linear_evolve(epoch, startEpoch, nEpochs, startVal, finalVal, shiftFwd=False):
    # Evolve a value linearly from startVal to finalVal across nEpochs
    # This formula returns epoch=startEpoch returns currVal<=startVal and epoch=startEpoch+nEpochs returns currVal=finalVal
    # Unfortunately, this implies that after nEpochs of evolution, the value has not yet reached finalVal
    # e.g., startEpoch=0, nEpochs=10: currVal=/=finalVal when epoch=9 (the 10th epoch), currVal=finalVal when epoch=10 (the 11th epoch)
    # This convention is normal in other SSL codes (e.g., SimSiam)
    # shiftFwd shifts the evolution trend by 1 epoch, such that the value reaches finalVal 1 epoch earlier

    if shiftFwd:
        epoch += 1

    if epoch <= startEpoch:
        return startVal
    elif epoch >= startEpoch + nEpochs:
        return finalVal
    else:
        return startVal + (finalVal - startVal) * (epoch - startEpoch) / nEpochs

def step_evolve(epoch, startEpoch, evolveFactor, evolveSteps, startVal, shiftFwd=False):
    # Evolves a value by multiplying by a factor at specific epochs
    # shiftFwd shifts the step by 1 epoch, such that multiplying happens on the ith epoch, rather than after it

    if shiftFwd:
        epoch += 1

    return startVal * evolveFactor ** sum([(epoch - startEpoch) >= step for step in evolveSteps])

def overwrite_ssl_args(args):
    """
    Library of default model settings for each SSL model
    This function should remain limited to settings that are vital to the SSL model identity, otherwise it will be a mess to set hyperparameters
    This function is incomplete, it requires more models to have their settings
    """

    if args.modelType == 'simsiam':
        assert args.prdHidDim > 0
        args.momEncBeta = 0.0
        args.applySG = True
        args.lossType = 'wince'
        args.winceBeta = 0.0

    elif args.modelType == 'byol':
        assert args.prdHidDim > 0
        assert args.momEncBeta > 0.0
        args.applySG = True
        args.lossType = 'wince'
        args.winceBeta = 0.0

    elif args.modelType == 'simclr':
        args.prdHidDim = 0
        args.momEncBeta = 0.0
        args.applySG = False
        args.lossType = 'wince'
        assert args.winceBeta > 0.0

    elif args.modelType == 'moco':
        assert args.prdHidDim > 0  # MoCoV3 uses a predictor
        assert args.momEncBeta > 0.0
        args.applySG = True  # MoCoV3 uses stop-gradient
        args.lossType = 'wince'
        assert args.winceBeta > 0.0

    return args
