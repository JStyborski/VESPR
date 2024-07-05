import argparse
import builtins
from distutils.util import strtobool
import os
import time

import torch
from torch import nn
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets

from Utils import Models
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
    parser.add_argument('--gpu', default=0, type=int, help='GPU ID to use for training (single GPU)')
    parser.add_argument('--nodeCount', default=1, type=int, help='Number of nodes/servers to use for distributed training')
    parser.add_argument('--nodeRank', default=0, type=int, help='Global rank of nodes/servers')
    parser.add_argument('--distURL', default='tcp://127.0.0.1:2345', type=str, help='URL for distributed training setup')
    parser.add_argument('--distBackend', default='nccl', type=str, help='Distributed backend method')
    parser.add_argument('--workersPerProc', default=4, type=int, help='Number of data-loading workers per process')
    parser.add_argument('--pinMem', default=True, type=lambda x: bool(strtobool(x)), help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU')
    parser.add_argument('--randSeed', default=None, type=int, help='RNG initial set point')

    # Dataset parameters
    parser.add_argument('--ptFile', default='', type=str, help='Single pretrained model to train - if empty, finetunes all in ptDir')
    parser.add_argument('--ptDir', default='Trained_Models', type=str, help='Folder containing pretrained models')
    parser.add_argument('--trainRoot', default='', type=str, help='Training dataset root directory')
    parser.add_argument('--filePrefix', default='', type=str, help='Prefix to add to finetuned file name')
    parser.add_argument('--ftType', default='lp', type=str, choices=['lp', 'ft'], help='Type of finetuning to apply - linear probe or finetune')
    parser.add_argument('--cropSize', default=224, type=int, help='Crop size to use for input images')

    # Training from scratch
    parser.add_argument('--ftFromNull', default=False, type=lambda x:bool(strtobool(x)), help='Boolean to train a model from scratch - ftType must be ft')
    parser.add_argument('--nullEncArch', default='resnet18', type=str, choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'vit_tiny', 'vit_small', 'vit_base', 'vit_large'], help='Encoder network (backbone) type')
    parser.add_argument('--nullRnCifarMod', default=False, type=lambda x:bool(strtobool(x)), help='Boolean to apply CIFAR modification to ResNets')
    parser.add_argument('--nullVitPosEmb', default='normal', type=str, choices=['normal', 'sincos'], help='Position embedding initialization method')
    parser.add_argument('--nullVitPPFreeze', default=True, type=lambda x:bool(strtobool(x)), help='Boolean to freeze ViT patch projection for training stability')

    # Training parameters
    parser.add_argument('--nEpochs', default=100, type=int, help='Number of epochs to run')
    parser.add_argument('--batchSize', default=512, type=int, help='Data loader batch size')
    parser.add_argument('--nBatches', default=1e10, type=int, help='Maximum number of batches to run per epoch')
    parser.add_argument('--momentum', default=0.9, type=float, help='SGD momentum value')
    parser.add_argument('--weightDecay', default=0, type=float, help='SGD weight decay value')
    parser.add_argument('--initLR', default=5.0, type=float, help='SGD initial learning rate')
    parser.add_argument('--useAMP', default=True, type=lambda x: bool(strtobool(x)), help='Boolean to apply AMP and loss scaling')
    parser.add_argument('--useLARS', default=False, type=lambda x:bool(strtobool(x)), help='Boolean to apply LARS optimizer')
    parser.add_argument('--lrWarmupEp', default=10, type=int, help='Number of linear warmup steps to apply on learning rate - set as 0 for no warmup')
    parser.add_argument('--decayLR', default='stepdn', type=str, choices=['stepdn', 'cosdn'], help='Learning rate decay method')
    parser.add_argument('--decaySteps', default=[60, 75, 90], type=list, help='Steps at which to apply stepdn decay')
    parser.add_argument('--decayFactor', default=0.2, type=float, help='Factor by which to multiply LR at step down')

    # Adversarial training parameters
    parser.add_argument('--useAdv', default=False, type=lambda x:bool(strtobool(x)), help='Boolean to apply adversarial training')
    parser.add_argument('--keepStd', default=False, type=lambda x:bool(strtobool(x)), help='Boolean to train with the adversarial plus original images - increases batch size')
    parser.add_argument('--advAlpha', default=0.6/255 / 0.226, type=float, help='PGD step size')
    parser.add_argument('--advEps', default=4/255 / 0.226, type=float, help='PGD attack radius limit, measured in specified norm')
    parser.add_argument('--advNorm', default=float('inf'), type=float, help='Norm type for measuring perturbation radius')
    parser.add_argument('--advRestarts', default=1, type=int, help='Number of PGD restarts to search for best attack')
    parser.add_argument('--advSteps', default=10, type=int, help='Number of PGD steps to take')
    parser.add_argument('--advBatchSize', default=512, type=int, help='Batch size to use for adversarial training loader')
    parser.add_argument('--advNoise', default=True, type=lambda x:bool(strtobool(x)), help='Boolean to use random initialization')
    parser.add_argument('--advNoiseMag', default=None, type=float, help='Magnitude of noise to add to random start attack')
    parser.add_argument('--advClipMin', default=None, type=float, help='Minimium value to clip adversarial inputs')
    parser.add_argument('--advClipMax', default=None, type=float, help='Maximum value to clip adversarial inputs')

    return parser

###################
# Setup Functions #
###################

def init_data_loader_sampler(args):

    print('Defining dataset and loader')

    trainDataset = datasets.ImageFolder(args.trainRoot, CT.t_finetune(args.cropSize))
    if args.useDDP:
        trainSampler = torch.utils.data.distributed.DistributedSampler(trainDataset)
        # When using single GPU per process and per DDP, need to divide batch size based on nGPUs
        args.batchSize = int(args.batchSize / args.nProcs)
        args.advBatchSize = int(args.advBatchSize / args.nProcs)
    else:
        trainSampler = None

    # Note that DistributedSampler automatically shuffles dataset given the set_epoch() function during training
    trainLoader = torch.utils.data.DataLoader(trainDataset, batch_size=args.batchSize, shuffle=(trainSampler is None),
                                              num_workers=args.workersPerProc, pin_memory=args.pinMem, sampler=trainSampler, drop_last=True)

    args.nClasses = len(trainDataset.classes)

    return trainLoader, trainSampler

def init_model(args, ptFile):

    print('- Instantiating model')

    # If training model from scratch, generate model with hacked projector and predictor
    if args.ftFromNull:

        # Create model
        args.encArch, args.rnCifarMod, args.vitPosEmb, args.vitPPFreeze = args.nullEncArch, args.nullRnCifarMod, args.nullVitPosEmb, args.nullVitPPFreeze
        model = Models.Base_Model(args.encArch, args.rnCifarMod, args.vitPosEmb, args.vitPPFreeze, 'moco', 2, 2, 2, 0, None, 0.3, 0.5, 0.0, True)

    # If model was pretrained, create model and load pretrain weights
    else:

        # Get state dict and create model
        SD = torch.load(ptFile, map_location='cuda:{}'.format(args.gpu))
        args.encArch, args.rnCifarMod, args.vitPosEmb, args.vitPPFreeze = SD['pt_args']['encArch'], SD['pt_args']['rnCifarMod'], SD['pt_args']['vitPosEmb'], SD['pt_args']['vitPPFreeze']
        model = Models.Base_Model(args.encArch, args.rnCifarMod, args.vitPosEmb, args.vitPPFreeze, SD['pt_args']['prjArch'], SD['pt_args']['prjHidDim'],
                                  SD['pt_args']['prjBotDim'], SD['pt_args']['prjOutDim'], SD['pt_args']['prdHidDim'], None, 0.3, 0.5, 0.0, True)

        print('- Loading pretrained model weights')
        model.load_state_dict(SD['modelSD'], strict=False)
        del SD # Delete state dict to save space

    # If using linear probe, freeze all layers (though predictor will later be replaced and trainable)
    if args.ftType == 'lp':
        for param in model.parameters(): param.requires_grad = False

    # Replace the projector with identity and the predictor with linear classifier
    model.projector = nn.Identity()
    model.predictor = nn.Linear(model.encDim, args.nClasses)

    print('- Setting up model on single/multiple devices')
    if args.useDDP:
        # Convert BN to SyncBN
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        # For multiproc distributed, DDP constructor should set the single device scope - otherwises uses all available
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

    # If linear probe, set model to eval mode (no BN updates). If finetuning, allow model to update
    model.eval() if args.ftType == 'lp' else model.train()

    return model

def init_optimizer(args, model):

    print('- Instantiating optimizer')

    if 'vit' in args.encArch.lower() and args.useDDP:
        optimizer = torch.optim.AdamW(params=model.module.parameters(), lr=args.initLR, weight_decay=args.weightDecay)
    elif 'vit' in args.encArch.lower():
        optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.initLR, weight_decay=args.weightDecay)
    elif args.useDDP:
        optimizer = torch.optim.SGD(params=model.module.parameters(), lr=args.initLR, momentum=args.momentum, weight_decay=args.weightDecay)
    else:
        optimizer = torch.optim.SGD(params=model.parameters(), lr=args.initLR, momentum=args.momentum, weight_decay=args.weightDecay)
    if args.useLARS:
        print("- Using LARS optimizer.")
        optimizer = MF.LARC(optimizer=optimizer, trust_coefficient=0.001, clip=False)

    return optimizer

#####################
# Running Functions #
#####################

def adjust_lr_per_epoch(args, epoch, optimizer):

    # Linear LR warmup applies to all layers
    if epoch <= args.lrWarmupEp:
        curLR = MF.linear_evolve(epoch, 1, args.lrWarmupEp, 0., args.initLR, shiftFwd=True)
        for param_group in optimizer.param_groups:
            param_group['lr'] = curLR

    # If desired, adjust learning rate for the current epoch
    if args.decayLR == 'stepdn' and epoch >= args.lrWarmupEp:
        curLR = MF.step_evolve(epoch, 1, args.decayFactor, args.decaySteps, args.initLR, shiftFwd=True)
        for param_group in optimizer.param_groups:
            param_group['lr'] = curLR
    elif args.decayLR == 'cosdn' and epoch >= args.lrWarmupEp:
        curLR = MF.cosine_evolve(epoch, 1, args.nEpochs, args.initLR, 0., shiftFwd=False)
        for param_group in optimizer.param_groups:
            param_group['lr'] = curLR

def train_one_batch(args, batch, model, lossFn, lossScaler, optimizer):

    # Get inputs, truth labels
    augTens = batch[0].cuda(args.gpu, non_blocking=True)
    truthTens = batch[1].cuda(args.gpu, non_blocking=True)

    # Calculate (untargeted) adversarial samples from inputs and use for training
    if args.useAdv:
        _, advTens = FGSM_PGD.sl_pgd(model, nn.CrossEntropyLoss(reduction='none'), args.useAMP, augTens, truthTens, args.advAlpha, args.advEps,
                                     args.advNorm, args.advRestarts, args.advSteps, args.advBatchSize, outIdx=0, targeted=False,
                                     randInit=args.advNoise, noiseMag=args.advNoiseMag, xMin=args.advClipMin, xMax=args.advClipMax)
        if args.keepStd:
            augTens = torch.cat((augTens, advTens.detach()), dim=0).cuda(args.gpu, non_blocking=True)
        else:
            augTens = advTens.detach()

    with torch.cuda.amp.autocast(enabled=args.useAMP):

        # Run augmented data through SSL model with linear classifier
        p, _, _, _ = model(augTens)

        # Calculate loss
        lossVal = lossFn(p, truthTens)

    optimizer.zero_grad()
    if args.useAMP:
        lossScaler.scale(lossVal).backward()  # Backprop on upscaled loss to prevent underflow with AMP
        # lossScaler.unscale_(optimizer)  # Optionally unscale gradients in order to clip or measure them
        lossScaler.step(optimizer)  # Loss scaler unscales gradients and steps weights
        lossScaler.update()
    else:
        lossVal.backward()
        optimizer.step()

    # Accuracy calculations
    nCorrect = torch.sum(torch.argmax(p.detach(), dim=1) == truthTens).cpu().numpy()
    nTotal = len(truthTens)

    return lossVal.detach(), nCorrect, nTotal

def save_checkpoint(args, ptFile, epoch, model, lossScaler, optimizer):

    state = {
        'ft_args'       : vars(args),
        'epoch'         : epoch,
        'modelSD'       : model.module.state_dict() if args.useDDP else model.state_dict(),
        'scalerSD'      : lossScaler.state_dict() if args.useAMP else None,
        'optimSD'       : optimizer.state_dict(),
    }
    chkptPath = ptFile[:-8] + '_' + args.filePrefix + '_' + args.ftType + '_{:04d}'.format(epoch) + '.pth.tar'
    if args.gpu == 0:
        torch.save(state, chkptPath)
    print(f'Saved Checkpoint: {chkptPath}')

#######################
# Execution Functions #
#######################

def ssl_finetune_main():

    argsParser = get_args_parser()
    args = argsParser.parse_args()

    if args.randSeed is not None:
        torch.manual_seed(args.randSeed)
        cudnn.deterministic = True
        print('You have chosen to seed training'
              'This will turn on the CUDNN deterministic setting, which can slow down your training'
              'You may see unexpected behavior when restarting from checkpoints')

    if not args.useDDP:
        print('You have disabled DDP - the model will train on 1 GPU without data parallelism')

    if args.ftFromNull:
        if not os.path.exists('Trained_Models'):
            os.mkdir('Trained_Models')
        # Finetune from scratch, create a fake pretrain filename
        args.ptList = [args.ptDir + '/' + args.filePrefix + '_pt_0000.pth.tar']
        assert args.ftType == 'ft'
    elif args.ptFile != '':
        # Single pretrain file
        args.ptList = [args.ptDir + '/' + args.ptFile]
    else:
        # List of pretrained models as any model with _pt_ in the name
        args.ptList = sorted([args.ptDir + '/' + stateFile for stateFile in os.listdir(args.ptDir)
                              if ('_pt_' in stateFile and '_lp_' not in stateFile and '_ft_' not in stateFile)])

    # Infer learning rate
    args.initLR = args.initLR * args.batchSize / 256

    # Launch multiple (or single) distributed processes for main_worker function - will automatically assign GPUs
    if args.useDDP:
        args.nProcPerNode = torch.cuda.device_count()
        args.nProcs = args.nProcPerNode * args.nodeCount
        torch.multiprocessing.spawn(ssl_finetune_worker, nprocs=args.nProcs, args=(args,))
    # Launch one process for main_worker function
    else:
        ssl_finetune_worker(args.gpu, args)

def ssl_finetune_worker(gpu, args):

    # Replace initial GPU index with assigned one
    args.gpu = gpu
    print('GPU {} online'.format(args.gpu))

    # Suppress printing if not master (gpu 0)
    if args.useDDP and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    print('Using GPU {} as master process'.format(args.gpu))

    # Set up distributed training on backend
    # For multiprocessing distributed training, rank needs to be the global rank among all the processes
    if args.useDDP:
        args.procRank = args.nodeRank * args.nProcPerNode + args.gpu
        torch.distributed.init_process_group(backend=args.distBackend, init_method=args.distURL, world_size=args.nProcs, rank=args.procRank)
        torch.distributed.barrier()

    # Initialize the data loader and sampler
    trainLoader, trainSampler = init_data_loader_sampler(args)

    for ptFile in args.ptList:

        print('\nFinetuning ' + ptFile)

        # Create model, loss function, loss scaler, and optimizer
        model = init_model(args, ptFile)
        lossFn = nn.CrossEntropyLoss()
        lossScaler = torch.cuda.amp.GradScaler() if args.useAMP else None
        optimizer = init_optimizer(args, model)

        # Start timer
        timeStart = time.time()

        print('- Beginning training')
        for epoch in range(1, args.nEpochs + 1):

            # Update sampler with current epoch - required to ensure shuffling works across devices
            if args.useDDP:
                trainSampler.set_epoch(epoch)

            # Adjust learning rate based on current epoch
            adjust_lr_per_epoch(args, epoch, optimizer)

            for batchI, batch in enumerate(trainLoader):

                lossVal, nCorrect, nTotal = train_one_batch(args, batch, model, lossFn, lossScaler, optimizer)

                if batchI + 1 >= args.nBatches:
                    break

            if epoch == 1 or epoch % 10 == 0:
                print('Epoch: {} / {} | CurLR: {:0.4f} | Time: {:0.2f} | Last Loss: {:0.4f} | Last Trn Acc: {:0.4f}'
                      .format(epoch, args.nEpochs, optimizer.param_groups[0]['lr'], time.time() - timeStart, lossVal.detach(), nCorrect / nTotal))

            # Synchronize all processes after epoch
            if args.useDDP and args.nProcs > 1:
                torch.distributed.barrier()

        # Save out finetune model
        save_checkpoint(args, ptFile, epoch, model, lossScaler, optimizer)

if __name__ == '__main__':
    ssl_finetune_main()
