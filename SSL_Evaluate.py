import numpy as np
import os
import torch
from torch import nn
import torchvision.datasets as datasets

from Utils import Models
import Utils.Custom_Transforms as CT
import Utils.Custom_Probes as CP
from Adversarial import FGSM_PGD, Local_Lip

######################
# Analysis Functions #
######################

def make_rep_bank(device, loader, model, batchSize, nBankBatches, encDim):

    repBank = torch.zeros(nBankBatches * batchSize, encDim)
    labelBank = torch.zeros(nBankBatches * batchSize)

    for batchI, batch in enumerate(loader):

        # Augment data and get labels
        augTens = batch[0].to(device)
        truthTens = batch[1].to(device)

        # Get input encodings and write encodings + labels to bank
        r = model(augTens)[2]
        repBank[batchI * batchSize:(batchI + 1) * batchSize, :] = r.detach()
        labelBank[batchI * batchSize:(batchI + 1) * batchSize] = truthTens

        if batchI + 1 >= nBankBatches:
            break

    return repBank, labelBank

def calculate_smoothness(device, loader, model, advArgs, normOuts):

    # Get new batch and reset model grads
    batch = next(iter(loader))
    augTens = batch[0].to(device)

    # Calculate local smoothness
    avgRepLolip, _ = Local_Lip.maximize_local_lip(model, augTens, advArgs['advAlpha'], advArgs['advEps'], 1, np.inf, advArgs['advRestarts'],
                                                  advArgs['advSteps'], advArgs['advBatchSize'], outIdx=2, normOuts=normOuts,
                                                  randInit=advArgs['randInit'], noiseMag=advArgs['noiseMag'],
                                                  xMin=advArgs['xMin'], xMax=advArgs['xMax'])

    return avgRepLolip

def knn_vote(device, loader, model, knnBank, labelBank, k, knnTestBatches, simType='cos'):

    accCount = 0
    totalCount = 0
    for batchI, batch in enumerate(loader):

        # Get data and labels
        augTens = batch[0].to(device)
        truthTens = batch[1].to(device).detach().cpu()

        # Run augmented data through model, get output before linear classifier
        r = model(augTens)[2]
        r = r.detach().cpu()

        batchSize = r.size(0)
        totalCount += batchSize

        # Loop through each sample in batch size and test KNN
        for i in range(batchSize):

            # Get the count of labels corresponding to the k nearest training vectors
            if simType == 'cos':
                nearestKIdx = nn.functional.cosine_similarity(r[[i], :], knnBank, dim=1).argsort()[-k:]
            elif simType == 'mse':
                nearestKIdx = torch.mean(torch.square(r[[i], :] - knnBank), dim=1).argsort()[:k]
            uniqLabels, counts = np.unique(labelBank[nearestKIdx].numpy(), return_counts=True)

            # Check for ties between nearest labels:
            nModalLabels = len(np.where(counts == np.max(counts))[0])
            if nModalLabels == 1:
                modalLabel = uniqLabels[np.argsort(counts)[-1]]
            else:
                modalLabel = uniqLabels[np.argsort(counts)[-1 * (1 + np.random.choice(nModalLabels))]]

            # Check if KNN label is correct
            if modalLabel == truthTens[i].numpy():
                accCount += 1

        if batchI + 1 >= knnTestBatches:
            break

    return accCount / totalCount

def train_test_acc(device, loader, model, nBatches, outIdx):

    nCorrect = 0
    nTotal = 0

    for batchI, batch in enumerate(loader):

        # CenterCrop raw data
        augTens = batch[0].to(device)
        truthTens = batch[1].to(device)

        # Run augmented data through SimSiam with linear classifier
        preds = model(augTens)[outIdx]

        # Keep running sum of loss
        nCorrect += torch.sum(torch.argmax(preds.detach(), dim=1) == truthTens).cpu().numpy()
        nTotal += batch[0].size(0)

        if batchI + 1 >= nBatches:
            break

    clnAcc = nCorrect / nTotal

    return clnAcc

def adv_attack(device, loader, model, advArgs, outIdx):

    # Get new batch
    batch = next(iter(loader))
    augTens = batch[0].to(device)
    truthTens = batch[1].to(device)

    # Attack batch of images with FGSM or PGD and calculate accuracy
    _, advTens = FGSM_PGD.sl_pgd(model, nn.NLLLoss(reduction='none'), False, augTens, truthTens, advArgs['advAlpha'], advArgs['advEps'], np.inf,
                                 advArgs['advRestarts'], advArgs['advSteps'], advArgs['advBatchSize'], outIdx=outIdx, targeted=False,
                                 randInit=advArgs['randInit'], noiseMag=advArgs['noiseMag'], xMin=advArgs['xMin'], xMax=advArgs['xMax'])
    advAcc = torch.sum(torch.argmax(model(advTens)[0].detach(), dim=1) == truthTens).cpu().numpy() / advTens.size(0)

    return advAcc

#######################
# Execution Functions #
#######################

def ssl_evaluate_main():

    ###############
    # User Inputs #
    ###############

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data Parameters
    trainRoot = r'../Datasets/Poisoned_ImageNet/TAP_100/train'
    testRoot = r'../Datasets/ImageNet100/val'
    cropSize = 224
    batchSize = 512

    # Find pretrained models
    dirName = 'Trained_Models'
    evalPTOnly = False
    # Single file
    fileList = [dirName + '/MoCo_AP8_pt_0200_AP8_lp_0100.pth.tar']
    # All pretrain/linprobe files
    # fileList = sorted([dirName + '/' + stateFile for stateFile in os.listdir(dirName) if ('_pt_' in stateFile and '_lp_' in stateFile)])
    # All pretrain/finetune files
    # fileList = sorted([dirName + '/' + stateFile for stateFile in os.listdir(dirName) if ('_pt_' in stateFile and '_ft_' in stateFile)])
    # All files finetuned from scratch
    #fileList = sorted([dirName + '/' + stateFile for stateFile in os.listdir(dirName) if ('_pt_0000_' in stateFile and '_ft_' in stateFile)])

    # KNN parameters
    knnBankBatches = 20
    knnTestBatches = 5
    k = 20

    # LinCls parameters
    trainAccBatches = 1e6
    testAccBatches = 1e6

    # Adversarial parameters
    advArgs = {'advSamples': 1024, 'advBatchSize': 64, 'advAlpha': 0.6 / 255 / 0.226, 'advEps': 4 / 255 / 0.226,
               'advRestarts': 5, 'advSteps': 10, 'randInit': True, 'noiseMag': None, 'xMin': None, 'xMax': None}
    #advArgs = {'advSamples': 1024, 'advBatchSize': 64, 'advAlpha': 0.6 / 255, 'advEps': 4 / 255,
    #           'advRestarts': 5, 'advSteps': 10, 'randInit': True, 'noiseMag': None, 'xMin': 0., 'xMax': 1.}

    ##############
    # Data Setup #
    ##############

    # Create datasets and dataloaders
    trainDataset = datasets.ImageFolder(trainRoot, CT.t_finetune(cropSize))
    testDataset = datasets.ImageFolder(testRoot, CT.t_test(cropSize))

    knnTrainLoader = torch.utils.data.DataLoader(trainDataset, batch_size=batchSize, shuffle=True)
    knnTestLoader = torch.utils.data.DataLoader(testDataset, batch_size=batchSize, shuffle=True)
    linTrainLoader = torch.utils.data.DataLoader(trainDataset, batch_size=batchSize, shuffle=True)
    linTestLoader = torch.utils.data.DataLoader(testDataset, batch_size=batchSize, shuffle=True)
    atkLoader = torch.utils.data.DataLoader(testDataset, batch_size=advArgs['advSamples'], shuffle=True)

    nClasses = len(trainDataset.classes)

    ##############
    # Evaluation #
    ##############

    probes = CP.Evaluation_Probes()

    for stateFile in fileList:

        print('Evaluating ' + stateFile)

        # Load saved state
        SD = torch.load(stateFile, map_location='cuda:{}'.format(0))

        # If only evaluating pretrain models
        if evalPTOnly:

            # Create SSL model
            model = Models.Base_Model(SD['pt_args']['encArch'], SD['pt_args']['rnCifarMod'], SD['pt_args']['vitPosEmb'], SD['pt_args']['vitPPFreeze'],
                                      SD['pt_args']['prjArch'], SD['pt_args']['prjHidDim'], SD['pt_args']['prjBotDim'], SD['pt_args']['prjOutDim'],
                                      SD['pt_args']['prdHidDim'], None, 0.3, 0.5, 0.0, True)

            # Load state dict
            model.load_state_dict(SD['modelSD'], strict=False)

            # Replace the projector/predictor with identity to save compute
            model.projector = nn.Identity()
            model.predictor = nn.Linear(model.encDim, nClasses)

            # Freeze parameters, send model to device, and set to eval mode
            for param in model.parameters(): param.requires_grad = False
            model = model.to(device)
            model.eval()

            # Evaluate representation smoothness
            repLolip = calculate_smoothness(device, atkLoader, model, advArgs, normOuts=False)
            normRepLolip = calculate_smoothness(device, atkLoader, model, advArgs, normOuts=True)

            # Evaluate KNN accuracy of the model
            knnBank, knnLabelBank = make_rep_bank(device, knnTrainLoader, model, batchSize, knnBankBatches, model.encDim)
            cosKnnAcc = knn_vote(device, knnTestLoader, model, knnBank, knnLabelBank, k, knnTestBatches, simType='cos')
            mseKnnAcc = knn_vote(device, knnTestLoader, model, knnBank, knnLabelBank, k, knnTestBatches, simType='mse')
            print('Cos KNN Accuracy: {:0.4f}'.format(cosKnnAcc))
            print('MSE KNN Accuracy: {:0.4f}'.format(mseKnnAcc))

            trainAcc = None
            testAcc = None
            advAcc = None

        # Else evaluating finetune files
        else:

            # Create SSL model with hacked projector and no predictor
            model = Models.Base_Model(SD['ft_args']['encArch'], SD['ft_args']['rnCifarMod'], SD['ft_args']['vitPosEmb'], SD['ft_args']['vitPPFreeze'],
                                      'moco', 2, 2, 2, 0, None, 0.3, 0.5, 0.0, True)

            # Replace the projector with identity and the predictor with linear classifier
            model.projector = nn.Identity()
            model.predictor = nn.Linear(model.encDim, nClasses)

            # Load model weights
            model.load_state_dict(SD['modelSD'], strict=True)

            # Freeze parameters, send model to device, and set to eval mode
            for param in model.parameters(): param.requires_grad = False
            model = model.to(device)
            model.eval()

            # Evaluate representation smoothness
            repLolip = calculate_smoothness(device, atkLoader, model, advArgs, normOuts=False)
            normRepLolip = calculate_smoothness(device, atkLoader, model, advArgs, normOuts=True)

            # Evaluate KNN accuracy of the model
            knnBank, knnLabelBank = make_rep_bank(device, knnTrainLoader, model, batchSize, knnBankBatches, model.encDim)
            cosKnnAcc = knn_vote(device, knnTestLoader, model, knnBank, knnLabelBank, k, knnTestBatches, simType='cos')
            mseKnnAcc = knn_vote(device, knnTestLoader, model, knnBank, knnLabelBank, k, knnTestBatches, simType='mse')
            print('Cos KNN Accuracy: {:0.4f}'.format(cosKnnAcc))
            print('MSE KNN Accuracy: {:0.4f}'.format(mseKnnAcc))

            # Clean dataset accuracy
            trainAcc = train_test_acc(device, linTrainLoader, model, trainAccBatches, outIdx=0)
            print('Train Accuracy: {:0.4f}'.format(trainAcc))
            testAcc = train_test_acc(device, linTestLoader, model, testAccBatches, outIdx=0)
            print('Test Accuracy: {:0.4f}'.format(testAcc))

            # Evaluate adversarial accuracy
            advAcc = adv_attack(device, atkLoader, model, advArgs, outIdx=0)
            print('Adv Accuracy: {:0.4f}'.format(advAcc))

        # Update probes for each ftFile
        probes.update_probes(SD['epoch'], knnBank, repLolip, normRepLolip, cosKnnAcc, mseKnnAcc, trainAcc, testAcc, advAcc)
        #probes.update_probes(100, knnBank, repLolip, normRepLolip, cosKnnAcc, mseKnnAcc, trainAcc, testAcc, advAcc)

    probes.write_probes()
    #probes.plot_probes()

if __name__ == '__main__':
    ssl_evaluate_main()
