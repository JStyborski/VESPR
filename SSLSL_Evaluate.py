import os
import torch
from torch import nn
import torchvision.datasets as datasets

from Utils import Models
import Utils.Custom_Transforms as CT
import Utils.Custom_Probes as CP
from SSL_Evaluate import make_rep_bank, calculate_smoothness, knn_vote, train_test_acc, adv_attack

def sslsl_evaluate_main():

    ###############
    # User Inputs #
    ###############

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data parameters
    trainRoot = r'D:/Poisoned_ImageNet/UE_100/train'
    testRoot = r'D:/ImageNet100/val'
    cropSize = 224
    batchSize = 512

    # Find pretrained models
    dirName = 'Trained_Models'
    # Single file
    #fileList = [dirName + '/SimSiam_UE_ptft_0200.pth.tar']
    # All pretrain/finetune files
    fileList = sorted([dirName + '/' + stateFile for stateFile in os.listdir(dirName) if '_ptft_' in stateFile])

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

    #######################
    # Finetune Evaluation #
    #######################

    probes = CP.Evaluation_Probes()

    for stateFile in fileList:

        print('Evaluating ' + stateFile)

        # Load saved state
        SD = torch.load(stateFile, map_location='cuda:{}'.format(0))

        # Create model and load model weights
        model = Models.PTFT_Model(SD['ptft_args']['encArch'], SD['ptft_args']['rnCifarMod'], SD['ptft_args']['vitPosEmb'], SD['ptft_args']['vitPPFreeze'],
                                  SD['ptft_args']['prjArch'], SD['ptft_args']['prjHidDim'], SD['ptft_args']['prjBotDim'],
                                  SD['ptft_args']['prjOutDim'], SD['ptft_args']['prdHidDim'], None, 0.3, 0.5, 0.0, True, False, nClasses)

        # Load model weights
        model.load_state_dict(SD['modelSD'], strict=False)

        # Replace the projector and predictor with identity to save space/compute (they are unused)
        model.projector = nn.Identity()
        model.predictor = nn.Identity()

        # Freeze parameters, send model to device, and set to eval mode
        for param in model.parameters(): param.requires_grad = False
        model = model.to(device)
        model.eval()

        # Evaluate representation smoothness
        repLolip = calculate_smoothness(device, atkLoader, model, advArgs, normOuts=False)
        normRepLolip = calculate_smoothness(device, atkLoader, model, advArgs, normOuts=True)

        # Evaluate KNN accuracy of the model
        # Can reuse the representation bank already assembled
        knnBank, knnLabelBank = make_rep_bank(device, knnTrainLoader, model, batchSize, knnBankBatches, model.encDim)
        cosKnnAcc = knn_vote(device, knnTestLoader, model, knnBank, knnLabelBank, k, knnTestBatches, simType='cos')
        mseKnnAcc = knn_vote(device, knnTestLoader, model, knnBank, knnLabelBank, k, knnTestBatches, simType='mse')
        print('Cos KNN Accuracy: {:0.4f}'.format(cosKnnAcc))
        print('MSE KNN Accuracy: {:0.4f}'.format(mseKnnAcc))

        # Train dataset accuracy
        trainAcc = train_test_acc(device, linTrainLoader, model, trainAccBatches, outIdx=-1)
        print('Train Accuracy: {:0.4f}'.format(trainAcc))
        testAcc = train_test_acc(device, linTestLoader, model, testAccBatches, outIdx=-1)
        print('Test Accuracy: {:0.4f}'.format(testAcc))

        # Evaluate adversarial accuracy
        advAcc = adv_attack(device, atkLoader, model, advArgs, outIdx=-1)
        print('Adv Accuracy: {:0.4f}'.format(advAcc))

        # Update probes
        probes.update_probes(SD['epoch'], knnBank, repLolip, normRepLolip, cosKnnAcc, mseKnnAcc, trainAcc, testAcc, advAcc)

    probes.write_probes()
    # probes.plot_probes()

if __name__ == '__main__':
    sslsl_evaluate_main()