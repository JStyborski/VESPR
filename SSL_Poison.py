import argparse
import builtins
from distutils.util import strtobool
import numpy as np
import os
import time

import torch
import torch.backends.cudnn as cudnn

from SSL_Pretrain import init_model, init_ssl_loss, init_optimizer, load_checkpoint, adjust_lr_per_epoch, save_checkpoint
import Utils.Custom_Dataset as CD
import Utils.Custom_Transforms as CT
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
    parser.add_argument('--deltaRoot', default='', type=str, help='Poison delta tensor root directory')
    parser.add_argument('--poisonRoot', default='', type=str, help='Poison dataset output root directory')
    parser.add_argument('--trainLabels', default=True, type=lambda x:bool(strtobool(x)), help='Boolean if the training data is in label folders')
    parser.add_argument('--filePrefix', default='', type=str, help='Prefix to add to pretrained file name')
    parser.add_argument('--nAugs', default=2, type=int, help='Number of augmentations of cropSize to apply to each batch')
    parser.add_argument('--cropSize', default=224, type=int, help='Crop size to use for input images')
    parser.add_argument('--nAugs2', default=0, type=int, help='Number of augmentations of cropSize2 to apply to each batch')
    parser.add_argument('--cropSize2', default=96, type=int, help='Second crop size to use for input images')

    # Training parameters
    parser.add_argument('--nEpochs', default=100, type=int, help='Number of epochs to run')
    parser.add_argument('--batchSize', default=128, type=int, help='Data loader batch size')
    parser.add_argument('--nBatches', default=1e10, type=int, help='Maximum number of batches to run per epoch')
    parser.add_argument('--modelSteps', default=1e10, type=int, help='Number of model training steps to run per epoch')
    parser.add_argument('--poisonSteps', default=1e10, type=int, help='Number of poison training steps to run per epoch')
    parser.add_argument('--momentum', default=0.9, type=float, help='SGD momentum value')
    parser.add_argument('--weightDecay', default=1e-4, type=float, help='SGD weight decay value')
    parser.add_argument('--initLR', default=0.5, type=float, help='SGD initial learning rate')
    parser.add_argument('--useAMP', default=True, type=lambda x: bool(strtobool(x)), help='Boolean to apply AMP and loss scaling')
    parser.add_argument('--useLARS', default=False, type=lambda x:bool(strtobool(x)), help='Boolean to apply LARS optimizer')
    parser.add_argument('--lrWarmupEp', default=10, type=int, help='Number of linear warmup steps to apply on learning rate - set as 0 for no warmup')
    parser.add_argument('--decayEncLR', default=True, type=lambda x:bool(strtobool(x)), help='Boolean to apply cosine decay to encoder/projector learning rate')
    parser.add_argument('--decayPrdLR', default=False, type=lambda x:bool(strtobool(x)), help='Boolean to apply cosine decay to predictor learning rate')
    parser.add_argument('--loadChkPt', default=None, type=str, help='File name of checkpoint from which to resume')

    # Model parameters
    parser.add_argument('--encArch', default='resnet18', type=str, choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'vit_tiny', 'vit_small', 'vit_base', 'vit_large'], help='Encoder network (backbone) type')
    parser.add_argument('--rnCifarMod', default=False, type=lambda x:bool(strtobool(x)), help='Boolean to apply CIFAR modification to ResNets')
    parser.add_argument('--vitPosEmb', default='normal', type=str, choices=['normal', 'sincos'], help='Position embedding initialization method')
    parser.add_argument('--vitPPFreeze', default=True, type=lambda x:bool(strtobool(x)), help='Boolean to freeze ViT patch projection for training stability')
    parser.add_argument('--modelType', default='custom', type=str, choices=['simsiam', 'simclr', 'mec', 'moco', 'byol', 'bt', 'vicreg', 'dino'], help='Option to overwrite inputs with pre-designated model defaults - set as custom for no overwrite')
    parser.add_argument('--prjArch', default='moco', type=str, choices=['simsiam', 'simclr', 'mec', 'moco', 'byol', 'barlow_twins', 'vicreg', 'dino_cnn', 'dino_vit'], help='Projector network type')
    parser.add_argument('--prjHidDim', default=2048, type=int, help='Projector hidden dimension')
    parser.add_argument('--prjBotDim', default=256, type=int, help='Projector bottleneck dimension (only used with DINO projector)')
    parser.add_argument('--prjOutDim', default=2048, type=int, help='Projector output dimension')
    parser.add_argument('--prdHidDim', default=0, type=int, help='Predictor hidden dimension - set as 0 for no predictor')
    parser.add_argument('--prdAlpha', default=None, type=float, help='Optimal predictor correlation exponent - set as None for no optimal predictor')
    parser.add_argument('--prdEps', default=0.3, type=float, help='Optimal predictor regularization coefficient')
    parser.add_argument('--prdBeta', default=0.5, type=float, help='Optimal predictor correlation update momentum')
    parser.add_argument('--momEncBeta', default=0.0, type=float, help='Momentum encoder update momentum - set as 0.0 for no momentum encoder')
    parser.add_argument('--applySG', default=False, type=lambda x:bool(strtobool(x)), help='Boolean to apply stop-gradient to one branch')
    parser.add_argument('--dualStream', default=False, type=lambda x:bool(strtobool(x)), help='Boolean to backpropagate loss through all SSL branches during sample training')

    # Loss parameters
    parser.add_argument('--symmetrizeLoss', default=True, type=lambda x:bool(strtobool(x)), help='Boolean to apply loss function equally on both augmentation batches')
    parser.add_argument('--lossType', default='wince', type=str, choices=['wince', 'bt', 'vicreg', 'mec', 'dino'], help='SSL loss type to apply')
    parser.add_argument('--winceBeta', default=1.0, type=float, help='Contrastive term coefficient in InfoNCE loss - set as 0.0 for no contrastive term')
    parser.add_argument('--winceTau', default=0.2, type=float, help='Contrastive loss temperature factor')
    parser.add_argument('--winceEps', default=0.0, type=float, help='Similarity perturbation constant for disentanglement - 0.0 applies no modification')
    parser.add_argument('--winceSameView', default='offdiag', type=str, choices=['none', 'offdiag', 'all'], help='Option for including same-view terms in CL. offdiag = SimCLR, none = MoCo')
    parser.add_argument('--btLam', default=0.005, type=float, help='Coefficient to apply to off-diagonal terms of BT loss')
    parser.add_argument('--btLossType', default='bt', type=str, help='Method of calculating loss for off-diagonal terms')
    parser.add_argument('--btNormType', default='bn', type=str, help='Method of normalizing encoding data')
    parser.add_argument('--vicAlpha', default=25.0, type=float, help='Coefficient on variance loss term')
    parser.add_argument('--vicBeta', default=25.0, type=float, help='Coefficient on invariance loss term')
    parser.add_argument('--vicGamma', default=1.0, type=float, help='Coefficient on covariance loss term')
    parser.add_argument('--mecEd2', default=0.06, type=float, help='Related to the coefficient applied to correlation matrix')
    parser.add_argument('--mecTaylorTerms', default=2, type=int, help='Number of Taylor expansion terms to include in matrix logarithm approximation')
    parser.add_argument('--dinoCentMom', default=0.9, type=float, help='Momentum coefficient for teacher center vector')
    parser.add_argument('--dinoTauS', default=0.1, type=float, help='Temperature for student network (online) softmax')
    parser.add_argument('--dinoTauT', default=0.05, type=float, help='Temperature for teacher network (target) softmax')

    # Adversarial training parameters
    parser.add_argument('--advAlpha', default=0.1, type=float, help='PGD step size')
    parser.add_argument('--advEps', default=8./255., type=float, help='PGD attack radius limit, measured in specified norm')
    parser.add_argument('--advNorm', default=float('inf'), type=float, help='Norm type for measuring perturbation radius')
    parser.add_argument('--advRestarts', default=1, type=int, help='Number of PGD restarts to search for best attack')
    parser.add_argument('--advSteps', default=5, type=int, help='Number of PGD steps to take')
    parser.add_argument('--advNoise', default=True, type=lambda x:bool(strtobool(x)), help='Boolean to use random initialization')
    parser.add_argument('--advClipMin', default=0., type=int, help='Minimium value to clip adversarial inputs')
    parser.add_argument('--advClipMax', default=1., type=int, help='Maximum value to clip adversarial inputs')
    parser.add_argument('--advSavePrecision', default=torch.float16, type=torch.dtype, help='Precision for saving poison deltas')

    return parser

###################
# Setup Functions #
###################

def init_data_loader_sampler(args):

    print('- Defining dataset and loader')

    if args.trainLabels:
        deltaDataset = CD.DatasetFolder_Plus_Path(args.deltaRoot, transform=None, loader=torch.load, extensions='')
        trainDataset = CD.ImageFolder_Plus_Poison(args.trainRoot, CT.t_tensor(), deltaDataset)
    else:
        deltaDataset = CD.No_Labels_Plus_Path(args.deltaRoot, transform=None, loader=torch.load)
        trainDataset = CD.No_Labels_Images_Plus_Poison(args.trainRoot, CT.t_tensor(), deltaDataset)

    if args.useDDP:
        trainSampler = torch.utils.data.distributed.DistributedSampler(trainDataset)
        # When using single GPU per process and per DDP, need to divide batch size based on nGPUs
        args.batchSize = int(args.batchSize / args.nProcs)
    else:
        trainSampler = None

    # Note that DistributedSampler automatically shuffles dataset given the set_epoch() function during training
    trainDataLoader = torch.utils.data.DataLoader(trainDataset, batch_size=args.batchSize, shuffle=(trainSampler is None), collate_fn=MF.collate_list,
                                                  num_workers=args.workersPerProc, pin_memory=args.pinMem, sampler=trainSampler, drop_last=True)

    return trainDataLoader, trainSampler

#####################
# Running Functions #
#####################

def train_model_one_batch(args, images, deltas, augTransform, model, lossFn, lossScaler, optimizer):

    # Combine images and poisons and augment
    augList = [[] for _ in range(args.nAugs)]
    for i in range(len(deltas)):
        poisonAugs = augTransform(torch.clamp(images[i] + args.advEps * deltas[i], min=args.advClipMin, max=args.advClipMax))
        for j in range(args.nAugs):
            augList[j].append(poisonAugs[j])

    # Collect augmentations as torch tensors
    for i in range(len(augList)):
        augList[i] = torch.stack(augList[i], dim=0).cuda(args.gpu, non_blocking=True)

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
    optimizer.zero_grad()
    if args.useAMP:
        lossScaler.scale(lossVal).backward()  # Backprop on upscaled loss to prevent underflow with AMP
        # lossScaler.unscale_(optimizer)  # Optionally unscale gradients in order to clip or measure them
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

    return lossVal.detach()

def train_deltas_one_batch(args, images, deltas, deltaPaths, augTransform, model, lossFn):

    for _ in range(args.advSteps):

        # Combine images and poisons and augment
        augList = [[] for _ in range(args.nAugs)]
        for i in range(len(deltas)):
            deltas[i].requires_grad = True
            poisonAugs = augTransform(torch.clamp(images[i] + args.advEps * deltas[i], min=args.advClipMin, max=args.advClipMax))
            for j in range(args.nAugs):
                augList[j].append(poisonAugs[j])

        # Collect augmentations as torch tensors
        for i in range(len(augList)):
            augList[i] = torch.stack(augList[i], dim=0).cuda(args.gpu, non_blocking=True)

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

            # Calculate loss and backpropagate
            lossVal = lossFn(outList)

        model.zero_grad()  # momenc and optPrd not included in optimizer - though they don't use grads, ensure grads through them are zeroed
        lossVal.backward()  # No loss scaling despite multi-precision, following results on CIFAR-10 and IN-100 (see comments in FGSM_PGD.sl_pgd)

        # Apply PGD attack
        for i in range(len(deltas)):
            eta = FGSM_PGD.optimize_linear(deltas[i].grad, None, norm=args.advNorm)
            deltas[i] = torch.clamp(deltas[i].data - args.advAlpha * eta, -1., 1.)

    # Save updated perturbations
    for i in range(len(deltas)):
        torch.save(deltas[i].cpu().type(args.advSavePrecision), deltaPaths[i])

    return lossVal.detach()

#######################
# Execution Functions #
#######################

def ssl_poison_main():

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

    # Initialize poison data directory (this will may a while, depending on trainRoot size)
    print('- Initializing Poison Deltas')
    CD.create_delta_tensors(args.trainRoot, args.deltaRoot, args.trainLabels, args.advNoise, args.advSavePrecision)

    # Infer learning rate
    args.initLR = args.initLR * args.batchSize / 256

    # Determine process settings
    args.nProcPerNode = torch.cuda.device_count()
    args.nProcs = args.nProcPerNode * args.nodeCount
    args.gatherTensors = (args.useDDP and args.nProcs > 1 and args.gatherTensors)

    # Launch multiple (or single) distributed processes for main_worker function - will automatically assign GPUs
    if args.useDDP or args.nProcs > 1:
        args.useDDP = True  # Ensure this flag is True, as multiple loops rely on it
        torch.multiprocessing.spawn(ssl_poison_worker, nprocs=args.nProcs, args=(args,))
    # Launch one process for main_worker function
    else:
        ssl_poison_worker(args.gpu, args)

    print('- Generating Poison Samples')
    CD.create_poison_images(args.trainRoot, args.deltaRoot, args.poisonRoot, args.trainLabels, args.advEps)

def ssl_poison_worker(gpu, args):

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

    # Define multiaug transform for after regular augmentation in dataset
    nAugTransform = CT.MultiCropTransform(args.nAugs, CT.t_tensor_aug(args.cropSize), args.nAugs2, CT.t_tensor_aug(args.cropSize2))

    # Load a checkpoint if available
    if args.loadChkPt is not None:
        load_checkpoint(args, model, lossScaler, optimizer)
    else:
        args.startEpoch = 1

    # Start timer
    timeStart = time.time()

    print('- Beginning training')
    for epoch in range(args.startEpoch, args.nEpochs + 1):

        # Update sampler with current epoch - required to ensure shuffling works across devices
        if args.useDDP:
            trainSampler.set_epoch(epoch)

        # Adjust learning rate based on current epoch
        adjust_lr_per_epoch(args, epoch, optimizer)

        # Setup for model training
        model.train()
        if args.useDDP and args.nProcs > 1:
            model.broadcast_buffers = True
        trainIterator1 = iter(trainDataLoader)

        # Loop through specified number of steps in loader
        for _ in range(min(args.modelSteps, len(trainDataLoader))):

            # Get next batch of inputs
            images, _, deltas, _ = next(trainIterator1)

            # Train model for given images/deltas
            lossVal = train_model_one_batch(args, images, deltas, nAugTransform, model, lossFn, lossScaler, optimizer)

        print('Epoch: {} / {} | CurLR: {:0.4f} | Time: {:0.2f} | Last Model Loss: {:0.4f}'
              .format(epoch, args.nEpochs, optimizer.param_groups[0]['lr'], time.time() - timeStart, lossVal.detach()))

        # Synchronize all processes after epoch
        if args.useDDP and args.nProcs > 1:
            torch.distributed.barrier()

        # Setup for poison training
        model.eval()
        if args.dualStream:
            model.applySG = False
        if args.useDDP and args.nProcs > 1:
            model.broadcast_buffers = False
        trainIterator2 = iter(trainDataLoader)

        # Loop through specified number of steps in loader
        for _ in range(min(args.poisonSteps, len(trainDataLoader))):

            # Get next batch of inputs
            images, _, deltas, deltaPaths = next(trainIterator2)

            # Train deltas for given images/deltas
            lossVal = train_deltas_one_batch(args, images, deltas, deltaPaths, nAugTransform, model, lossFn)

        print('Epoch: {} / {} | CurLR: {:0.4f} | Time: {:0.2f} | Last Poison Loss: {:0.4f}'
              .format(epoch, args.nEpochs, optimizer.param_groups[0]['lr'], time.time() - timeStart, lossVal.detach()))

        # Reset model SG to its original state
        if args.dualStream:
            model.applySG = args.applySG

        # Synchronize all processes after epoch
        if args.useDDP and args.nProcs > 1:
            torch.distributed.barrier()

        # Checkpoint model
        if epoch % 50 == 0:
            save_checkpoint(args, epoch, model, lossScaler, optimizer)

if __name__ == '__main__':
    ssl_poison_main()
