import argparse
import builtins
from distutils.util import strtobool
import numpy as np
import os
import time

import torch
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets

from Utils import Models
from Utils import Losses
import Utils.Custom_Dataset as CD
import Utils.Custom_Transforms as CT
import Utils.Custom_Probes as CP
import Utils.Misc_Functions as MF
from Adversarial import FGSM_PGD

# CUDNN automatically searches for the best algorithm for processing a given model/optimizer/dataset
cudnn.benchmark = True

###############
# User Inputs #
###############

def get_args_parser():

    parser = argparse.ArgumentParser()

    # Run processing parameters
    parser.add_argument('--useDDP', default=False, type=lambda x:bool(strtobool(x)), help='Strategy to launch on single/multiple GPU')
    parser.add_argument('--gatherTensors', default=True, type=lambda x:bool(strtobool(x)), help='Collect tensors across multiple GPU for loss and backpropagation')
    parser.add_argument('--gpu', default=0, type=int, help='GPU ID to use for training (single GPU)')
    parser.add_argument('--nodeCount', default=1, type=int, help='Number of nodes/servers to use for distributed training')
    parser.add_argument('--nodeRank', default=0, type=int, help='Global rank of nodes/servers')
    parser.add_argument('--distURL', default='tcp://127.0.0.1:2345', type=str, help='URL for distributed training setup')
    parser.add_argument('--distBackend', default='nccl', type=str, help='Distributed backend method')
    parser.add_argument('--workersPerProc', default=4, type=int, help='Number of data-loading workers per process')
    parser.add_argument('--pinMem', default=True, type=lambda x: bool(strtobool(x)), help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU')
    parser.add_argument('--randSeed', default=None, type=int, help='RNG initial set point')

    # Dataset parameters
    parser.add_argument('--trainRoot', default='', type=str, help='Training dataset root directory')
    parser.add_argument('--trainLabels', default=True, type=lambda x:bool(strtobool(x)), help='Boolean if the training data is in label folders')
    parser.add_argument('--filePrefix', default='', type=str, help='Prefix to add to pretrained file name')
    parser.add_argument('--nAugs', default=2, type=int, help='Number of augmentations of cropSize to apply to each batch')
    parser.add_argument('--cropSize', default=224, type=int, help='Crop size to use for input images')
    parser.add_argument('--nAugs2', default=0, type=int, help='Number of augmentations of cropSize2 to apply to each batch')
    parser.add_argument('--cropSize2', default=96, type=int, help='Second crop size to use for input images')

    # Training parameters
    parser.add_argument('--nEpochs', default=200, type=int, help='Number of epochs to run')
    parser.add_argument('--batchSize', default=128, type=int, help='Data loader batch size')
    parser.add_argument('--nBatches', default=1e10, type=int, help='Maximum number of batches to run per epoch')
    parser.add_argument('--momentum', default=0.9, type=float, help='SGD momentum value')
    parser.add_argument('--weightDecay', default=1e-4, type=float, help='SGD weight decay value')
    parser.add_argument('--initLR', default=0.5, type=float, help='SGD initial learning rate')
    parser.add_argument('--useAMP', default=True, type=lambda x:bool(strtobool(x)), help='Boolean to apply AMP and loss scaling')
    parser.add_argument('--useLARS', default=False, type=lambda x:bool(strtobool(x)), help='Boolean to apply LARS optimizer')
    parser.add_argument('--lrWarmupEp', default=10, type=int, help='Number of linear warmup steps to apply on learning rate - set as 0 for no warmup')
    parser.add_argument('--decayEncLR', default=True, type=lambda x:bool(strtobool(x)), help='Boolean to apply cosine decay to encoder/projector learning rate')
    parser.add_argument('--decayPrdLR', default=False, type=lambda x:bool(strtobool(x)), help='Boolean to apply cosine decay to predictor learning rate')
    parser.add_argument('--loadChkPt', default=None, type=str, help='File name of checkpoint from which to resume')
    parser.add_argument('--runProbes', default=True, type=lambda x:bool(strtobool(x)), help='Boolean to run metric probes')

    # Model parameters
    parser.add_argument('--encArch', default='resnet18', type=str, choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'vit_tiny', 'vit_small', 'vit_base', 'vit_large'], help='Encoder network (backbone) type')
    parser.add_argument('--rnCifarMod', default=False, type=lambda x:bool(strtobool(x)), help='Boolean to apply CIFAR modification to ResNets')
    parser.add_argument('--vitPosEmb', default='normal', type=str, choices=['normal', 'sincos'], help='Position embedding initialization method')
    parser.add_argument('--vitPPFreeze', default=True, type=lambda x:bool(strtobool(x)), help='Boolean to freeze ViT patch projection for training stability')
    parser.add_argument('--modelType', default='custom', type=str, choices=['simsiam', 'simclr', 'mec', 'moco', 'byol', 'bt', 'vicreg', 'dino'], help='Option to overwrite inputs with pre-designated model defaults - set as custom for no overwrite')
    parser.add_argument('--prjArch', default='moco', type=str, choices=['simsiam', 'simclr', 'mec', 'moco', 'byol', 'bt', 'vicreg', 'dino_cnn', 'dino_vit'], help='Projector network type')
    parser.add_argument('--prjHidDim', default=2048, type=int, help='Projector hidden dimension')
    parser.add_argument('--prjBotDim', default=256, type=int, help='Projector bottleneck dimension (only used with DINO projector)')
    parser.add_argument('--prjOutDim', default=2048, type=int, help='Projector output dimension')
    parser.add_argument('--prdHidDim', default=0, type=int, help='Predictor hidden dimension - set as 0 for no predictor')
    parser.add_argument('--prdAlpha', default=None, type=float, help='Optimal predictor correlation exponent - set as None for no optimal predictor')
    parser.add_argument('--prdEps', default=0.3, type=float, help='Optimal predictor regularization coefficient')
    parser.add_argument('--prdBeta', default=0.5, type=float, help='Optimal predictor correlation update momentum')
    parser.add_argument('--momEncBeta', default=0.0, type=float, help='Momentum encoder update momentum - set as 0.0 for no momentum encoder')
    parser.add_argument('--applySG', default=False, type=lambda x:bool(strtobool(x)), help='Boolean to apply stop-gradient to one branch')

    # Loss parameters
    parser.add_argument('--symmetrizeLoss', default=True, type=lambda x:bool(strtobool(x)), help='Boolean to apply loss function equally on both augmentation batches')
    parser.add_argument('--lossType', default='wince', type=str, choices=['wince', 'bt', 'vicreg', 'mec', 'dino'], help='SSL loss type to apply')
    parser.add_argument('--winceBeta', default=1.0, type=float, help='Contrastive term coefficient in InfoNCE loss - set as 0.0 for no contrastive term')
    parser.add_argument('--winceTau', default=0.2, type=float, help='Contrastive loss temperature factor')
    parser.add_argument('--winceEps', default=0.0, type=float, help='Similarity perturbation constant for disentanglement - 0.0 applies no modification')
    parser.add_argument('--winceSameView', default='offdiag', type=str, choices=['none', 'offdiag', 'all'], help='Option for including same-view terms in CL. offdiag = SimCLR, none = MoCo')
    parser.add_argument('--btLam', default=0.005, type=float, help='Coefficient to apply to off-diagonal terms of BT loss')
    parser.add_argument('--btLossType', default='bt', type=str, choices=['bt', 'hsic'], help='Method of calculating loss for off-diagonal terms')
    parser.add_argument('--btNormType', default='bn', type=str, choices=['bn', 'l2'], help='Method of normalizing encoding data')
    parser.add_argument('--vicAlpha', default=25.0, type=float, help='Coefficient on variance loss term')
    parser.add_argument('--vicBeta', default=25.0, type=float, help='Coefficient on invariance loss term')
    parser.add_argument('--vicGamma', default=1.0, type=float, help='Coefficient on covariance loss term')
    parser.add_argument('--mecEd2', default=0.06, type=float, help='Related to the coefficient applied to correlation matrix')
    parser.add_argument('--mecTaylorTerms', default=2, type=int, help='Number of Taylor expansion terms to include in matrix logarithm approximation')
    parser.add_argument('--dinoCentMom', default=0.9, type=float, help='Momentum coefficient for teacher center vector')
    parser.add_argument('--dinoTauS', default=0.1, type=float, help='Temperature for student network (online) softmax')
    parser.add_argument('--dinoTauT', default=0.05, type=float, help='Temperature for teacher network (target) softmax')

    # Adversarial training parameters
    parser.add_argument('--useAdvList', default=[False, False], nargs='+', type=lambda x:bool(strtobool(x)), help='List of Booleans to apply adversarial training for each view')
    parser.add_argument('--keepStd', default=False, type=lambda x:bool(strtobool(x)), help='Boolean to train with the adversarial plus original images - increases batch size')
    parser.add_argument('--advAlpha', default=0.6/255 / 0.226, type=float, help='PGD step size')
    parser.add_argument('--advEps', default=4/255 / 0.226, type=float, help='PGD attack radius limit, measured in specified norm')
    parser.add_argument('--advNorm', default=float('inf'), type=float, help='Norm type for measuring perturbation radius')
    parser.add_argument('--advRestarts', default=1, type=int, help='Number of PGD restarts to search for best attack')
    parser.add_argument('--advSteps', default=10, type=int, help='Number of PGD steps to take')
    parser.add_argument('--advBatchSize', default=128, type=int, help='Batch size to use for adversarial training loader')
    parser.add_argument('--advNoise', default=True, type=lambda x:bool(strtobool(x)), help='Boolean to use random initialization')
    parser.add_argument('--advNoiseMag', default=None, type=float, help='Magnitude of noise to add to random start attack')
    parser.add_argument('--advClipMin', default=None, type=int, help='Minimium value to clip adversarial inputs')
    parser.add_argument('--advClipMax', default=None, type=int, help='Maximum value to clip adversarial inputs')

    return parser

###################
# Setup Functions #
###################

def init_data_loader_sampler(args):

    print('- Defining dataset and loader')

    if args.trainLabels:
        trainDataset = datasets.ImageFolder(args.trainRoot, CT.MultiCropTransform(args.nAugs, CT.t_pretrain(args.cropSize),
                                                                                  args.nAugs2, CT.t_pretrain(args.cropSize2)))
    else:
        trainDataset = CD.No_Labels_Images(args.trainRoot, CT.MultiCropTransform(args.nAugs, CT.t_pretrain(args.cropSize),
                                                                                 args.nAugs2, CT.t_pretrain(args.cropSize2)))

    if args.useDDP:
        trainSampler = torch.utils.data.distributed.DistributedSampler(trainDataset)
        # When using single GPU per process and per DDP, need to divide batch size based on nGPUs
        args.batchSize = int(args.batchSize / args.nProcs)
        args.advBatchSize = int(args.advBatchSize / args.nProcs)
    else:
        trainSampler = None

    # Note that DistributedSampler automatically shuffles dataset given the set_epoch() function during training
    trainDataLoader = torch.utils.data.DataLoader(trainDataset, batch_size=args.batchSize, shuffle=(trainSampler is None),
                                                  num_workers=args.workersPerProc, pin_memory=args.pinMem, sampler=trainSampler, drop_last=True)

    return trainDataLoader, trainSampler

def init_model(args):

    print('- Instantiating new model with {} backbone'.format(args.encArch))

    model = Models.Base_Model(args.encArch, args.rnCifarMod, args.vitPosEmb, args.vitPPFreeze, args.prjArch, args.prjHidDim, args.prjBotDim,
                              args.prjOutDim, args.prdHidDim, args.prdAlpha, args.prdEps, args.prdBeta, args.momEncBeta, args.applySG)

    # Set up model on parallel or single GPU
    if args.useDDP:
        # Convert BN to SyncBN
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        # For multiprocess distributed, DDP constructor should set the single device scope - otherwises uses all available
        torch.cuda.set_device(args.gpu)
        model.cuda(args.gpu)
        # broadcast_buffers (default=True) lets SyncBN sync running mean and variance for BN
        # There is a bug in the specific case of applying DDP to 1 GPU:
        # https://github.com/pytorch/pytorch/issues/73332, https://github.com/pytorch/pytorch/issues/66504
        # Workaround is to set broadcast_buffers = False when using 1 GPU
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], broadcast_buffers=(args.nProcs > 1))
    # Single GPU training
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        raise NotImplementedError('CPU training not supported')

    return model

def init_ssl_loss(args):

    print('- Instantiating loss functions')

    if args.lossType == 'wince':
        lossFn = Losses.Weighted_InfoNCE_Loss(args.symmetrizeLoss, args.winceBeta, args.winceTau, args.winceEps, args.winceSameView)
    elif args.lossType == 'bt':
        lossFn = Losses.Barlow_Twins_Loss(args.symmetrizeLoss, args.btLam, args.btLossType, args.btNormType)
    elif args.lossType == 'vicreg':
        lossFn = Losses.VICReg_Loss(args.symmetrizeLoss, args.vicAlpha, args.vicBeta, args.vicGamma)
    elif args.lossType == 'mec':
        lossFn = Losses.MEC_Loss(args.symmetrizeLoss, args.mecEd2, args.mecTaylorTerms)
    elif args.lossType == 'dino':
        lossFn = Losses.DINO_Loss(args.symmetrizeLoss, args.dinoCentMom, args.dinoTauS, args.dinoTauT)

    return lossFn

def init_optimizer(args, model):

    # Instantiate custom optimizer that skips momentum encoder and applies decay
    print('- Instantiating optimizer')

    if args.useDDP:
        optimParams = [{'params': model.module.encoder.parameters(), 'decayLR': args.decayEncLR},
                       {'params': model.module.projector.parameters(), 'decayLR': args.decayEncLR},
                       {'params': model.module.predictor.parameters(), 'decayLR': args.decayPrdLR}]
    else:
        optimParams = [{'params': model.encoder.parameters(), 'decayLR': args.decayEncLR},
                       {'params': model.projector.parameters(), 'decayLR': args.decayEncLR},
                       {'params': model.predictor.parameters(), 'decayLR': args.decayPrdLR}]

    if 'vit' in args.encArch.lower():
        optimizer = torch.optim.AdamW(params=optimParams, lr=args.initLR, weight_decay=args.weightDecay)
    else:
        optimizer = torch.optim.SGD(params=optimParams, lr=args.initLR, momentum=args.momentum, weight_decay=args.weightDecay)

    if args.useLARS:
        print('- Using LARS optimizer')
        optimizer = MF.LARC(optimizer=optimizer, trust_coefficient=0.001, clip=False)

    return optimizer

def load_checkpoint(args, model, lossScaler, optimizer):

    # Resume model/optimizer from checkpoint
    print('- Loading checkpoint file {}'.format(args.loadChkPt))

    # Map model to be loaded to specified single GPU
    chkPt = torch.load(args.loadChkPt, map_location='cuda:{}'.format(args.gpu))
    args.startEpoch = chkPt['epoch'] + 1  # Start at the next epoch after checkpoint save
    model.module.load_state_dict(chkPt['modelSD'], strict=True) if args.useDDP else model.load_state_dict(chkPt['modelSD'])
    lossScaler.load_state_dict(chkPt['scalerSD'])
    optimizer.load_state_dict(chkPt['optimSD'])
    del chkPt  # To save space

#####################
# Running Functions #
#####################

def adjust_lr_per_epoch(args, epoch, optimizer):

    # Linear LR warmup applies to all layers
    if epoch <= args.lrWarmupEp:
        curLR = MF.linear_evolve(epoch, 1, args.lrWarmupEp, 0., args.initLR, shiftFwd=True)
        for param_group in optimizer.param_groups:
            param_group['lr'] = curLR

    # Cosine annealing applied to modules with decayLR = True
    if (args.decayEncLR or args.decayPrdLR) and epoch >= args.lrWarmupEp:
        curLR = MF.cosine_evolve(epoch, 1, args.nEpochs, args.initLR, 0., shiftFwd=False)  # Decay during warmup
        #curLR = MF.cosine_evolve(epoch, args.lrWarmupEp - 1, args.nEpochs - args.lrWarmupEp, args.initLR, 0., shiftFwd=False) # Decay after warmup
        for param_group in optimizer.param_groups:
            if param_group['decayLR']:
                param_group['lr'] = curLR

def train_one_batch(args, batch, model, lossFn, lossScaler, optimizer):

    # Collect augmentations
    augList = [aug.cuda(args.gpu, non_blocking=True) for aug in batch[0]]

    # Run adversarial training
    if any(args.useAdvList):

        # Need to set broadcast_buffers to False for adversarial training with SSL
        # In the case of DDP with nProcs > 1, broadcast_buffers is true (sync BN on forward pass)
        # With adversarial attack, the model should be set to eval mode to not update parameters while finding adversarial examples
        # If you run a forward pass 2x with broadcast_buffers true, even for a model in eval mode, the hooks will try to broadcast
        # This broadcast is seen as an in-place operation, replacing original values and causing an error in the grad graph
        # Regular SSL is fine because the model is in train mode. SL AT is fine because there's only one forward pass
        # https://github.com/pytorch/pytorch/issues/22095, https://github.com/pytorch/pytorch/issues/66504, https://discuss.pytorch.org/t/regular-batchnorm-triggers-buffer-broadcast/137801
        if args.useDDP and args.nProcs > 1:
            model.broadcast_buffers = False

        _, advTensList = FGSM_PGD.ssl_pgd(model, lossFn, args.useAMP, augList, args.useAdvList, args.gatherTensors, args.advAlpha, args.advEps,
                                          args.advNorm, args.advRestarts, args.advSteps, args.advBatchSize, targeted=False,
                                          randInit=args.advNoise, noiseMag=args.advNoiseMag, xMin=args.advClipMin, xMax=args.advClipMax)

        # Reset model broadcast buffers setting if altered earlier
        if args.useDDP and args.nProcs > 1:
            model.broadcast_buffers = True

        # Optionally concatenate adversarial inputs with original inputs
        for i in range(len(advTensList)):
            if args.keepStd:
                augList[i] = torch.cat((augList[i], advTensList[i].detach()), dim=0).cuda(args.gpu, non_blocking=True)
            else:
                augList[i] = advTensList[i].detach()

    with torch.cuda.amp.autocast(enabled=args.useAMP):

        # Loop through each of the augs and create a list of results
        outList = []
        for augTens in augList:

            # Get input tensor, push through model
            p, z, r, mz = model(augTens)

            # Gather tensors across GPUs (required for accurate loss calculations)
            if args.gatherTensors:
                p = torch.cat(MF.FullGatherLayer.apply(p.contiguous()), dim=0)
                z = torch.cat(MF.FullGatherLayer.apply(z.contiguous()), dim=0)
                r = torch.cat(MF.FullGatherLayer.apply(r.contiguous()), dim=0)
                mz = torch.cat(MF.FullGatherLayer.apply(mz.contiguous()), dim=0)

            # Append to lists for loss calculation
            outList.append([p, z, r, mz])

        # Calculate loss
        lossVal = lossFn(outList)

    model.zero_grad()  # momenc and optPrd not included in optimizer - though they don't use grads, ensure grads through them are zeroed
    optimizer.zero_grad()  # Redundant with model.zero_grad()? I think so...
    if args.useAMP:
        lossScaler.scale(lossVal).backward()  # Backprop on upscaled loss to prevent underflow with AMP
        #lossScaler.unscale_(optimizer)  # Optionally unscale gradients in order to clip or measure them
        lossScaler.step(optimizer)  # Loss scaler unscales gradients and steps weights
        lossScaler.update()
    else:
        lossVal.backward()
        optimizer.step()

    # Update momentum encoder
    if args.momEncBeta > 0.0:
        if args.useDDP:
            model.module.update_momentum_network()
        else:
            model.update_momentum_network()

    return lossVal.detach(), outList

def save_checkpoint(args, epoch, model, lossScaler, optimizer):

    state = {
        'pt_args'       : vars(args),
        'epoch'         : epoch,
        'modelSD'       : model.module.state_dict() if args.useDDP else model.state_dict(),
        'scalerSD'      : lossScaler.state_dict() if args.useAMP else None,
        'optimSD'       : optimizer.state_dict(),
    }
    chkptPath = 'Trained_Models/{}_pt_{:04d}.pth.tar'.format(args.filePrefix, epoch)
    if args.gpu == 0:
        torch.save(state, chkptPath)
    print(f'Saved Checkpoint: {chkptPath}')

#######################
# Execution Functions #
#######################

def ssl_pretrain_main():

    argsParser = get_args_parser()
    args = argsParser.parse_args()
    if args.modelType != 'custom':
        args = MF.overwrite_ssl_args(args)

    if args.randSeed is not None:
        np.random.seed(args.randSeed)
        torch.manual_seed(args.randSeed)
        cudnn.deterministic = True
        print('You have chosen to seed training'
              'This will turn on the CUDNN deterministic setting, which can slow down your training'
              'You may see unexpected behavior when restarting from checkpoints')

    if not args.useDDP:
        print('You have disabled DDP - the model will train on 1 GPU without data parallelism')

    if not os.path.exists('Trained_Models'):
        os.mkdir('Trained_Models')

    # Infer learning rate
    args.initLR = args.initLR * args.batchSize / 256

    # Determine process settings
    args.nProcPerNode = torch.cuda.device_count()
    args.nProcs = args.nProcPerNode * args.nodeCount
    args.gatherTensors = (args.useDDP and args.nProcs > 1 and args.gatherTensors)

    # Launch multiple (or single) distributed processes for main_worker function - will automatically assign GPUs
    if args.useDDP or args.nProcs > 1:
        args.useDDP = True  # Ensure this flag is True, as multiple loops rely on it
        torch.multiprocessing.spawn(ssl_pretrain_worker, nprocs=args.nProcs, args=(args,))
    # Launch one process for main_worker function
    else:
        ssl_pretrain_worker(args.gpu, args)

def ssl_pretrain_worker(gpu, args):

    # Replace initial GPU index with assigned one
    args.gpu = gpu
    print('- GPU {} online'.format(args.gpu))

    # Suppress printing if not master (gpu 0)
    if args.useDDP and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass
    print('- Using GPU {} as master process'.format(args.gpu))

    # Set up distributed training on backend
    # For multiprocessing distributed training, rank needs to be the global rank among all the processes
    if args.useDDP:
        args.procRank = args.nodeRank * args.nProcPerNode + args.gpu
        torch.distributed.init_process_group(backend=args.distBackend, init_method=args.distURL, world_size=args.nProcs, rank=args.procRank)
        torch.distributed.barrier()

    # Define dataset, model, loss, and optimizer
    trainDataLoader, trainSampler = init_data_loader_sampler(args)
    model = init_model(args)
    lossFn = init_ssl_loss(args)
    lossScaler = torch.cuda.amp.GradScaler() if args.useAMP else None
    optimizer = init_optimizer(args, model)

    # Load a checkpoint if available
    if args.loadChkPt is not None:
        load_checkpoint(args, model, lossScaler, optimizer)
    else:
        args.startEpoch = 1

    # Initialize probes for training metrics and start timer
    if args.runProbes and args.gpu == 0:
        probes = CP.Pretrain_Probes()
    timeStart = time.time()

    print('- Beginning training')
    for epoch in range(args.startEpoch, args.nEpochs + 1):

        # Update sampler with current epoch - required to ensure shuffling works across devices
        if args.useDDP:
            trainSampler.set_epoch(epoch)

        # Adjust learning rate based on current epoch
        adjust_lr_per_epoch(args, epoch, optimizer)

        # Set model to train
        model.train()

        for batchI, batch in enumerate(trainDataLoader):

            # Train model for given batch
            lossVal, outList = train_one_batch(args, batch, model, lossFn, lossScaler, optimizer)

            # Stop training early if desired
            if batchI + 1 >= args.nBatches:
                break

        print('Epoch: {} / {} | CurLR: {:0.4f} | Time: {:0.2f} | Last Loss: {:0.4f}'
              .format(epoch, args.nEpochs, optimizer.param_groups[0]['lr'], time.time() - timeStart, lossVal.detach()))

        # Track record metrics while running
        if args.runProbes and args.gpu == 0:
            probes.update_probes(epoch, lossVal, None, None, outList)
            if epoch % 50 == 0:
                probes.write_probes(epoch, args.filePrefix)
                # probes.plot_probes()

        # Synchronize all processes after epoch and probe writing
        if args.useDDP and args.nProcs > 1:
            torch.distributed.barrier()

        # Checkpoint model
        if epoch % 50 == 0:
            save_checkpoint(args, epoch, model, lossScaler, optimizer)

if __name__ == '__main__':
    ssl_pretrain_main()
