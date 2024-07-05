# VESPR: Vulnerability Exploitation of Supervised Poisoning for Robust SSL

This library is a subset of the SSL_Sandbox library, a private in-development library for various SSL and SSL+SL techniques.
To promote readability and to preserve any competitive advantage I have, I've removed the in-development sections. 
I expect that the SSL_Sandbox library will eventually be made public and I will merge VESPR into it.
Instructions below mirror those of SSL_Sandbox.

This library was created to generate and train SSL models from multiple landmark methods, yet with highly customizeable architectures and training methods. 
Replicable models include the following:
- SimSiam (Chen and He, 2020: https://arxiv.org/abs/2011.10566)
- BYOL (Grill et al., 2020: https://arxiv.org/abs/2006.07733)
- SimCLR (Chen et al., 2020: https://arxiv.org/abs/2002.05709)
- MoCoV3 (Chen et al., 2021: https://arxiv.org/abs/2104.02057)
- Barlow Twins (Zbontar et al., 2021: https://arxiv.org/abs/2103.03230)
- VICReg (Bardes et al., 2021: https://arxiv.org/abs/2105.04906)
- MEC (Liu et al., 2022: https://arxiv.org/abs/2210.11464)
- DINO (Caron et al., 2021: https://arxiv.org/abs/2104.14294)

Models and algorithms are highly customizeable. The customizeable settings include the following:
- Online Encoder - Any torchvision ResNet and most TIMM ViTs
- Online Projector - Multiple options, mirroring the projectors used in the above SSL methods, with customizeable hidden layer and output layer sizes
- Asymmetric Predictor - MLP from BYOL, SimSiam, and MoCoV3 papers
- Target Encoder/Projector - Mimics online encoder/projector, optionally with EMA update and/or stop-gradient
- Loss Function - Multiple options mirroring the losses from above SSL methods, with customizeable parameters and optional loss-symmetrization
- Optimizer - ResNets default to SGD, ViTs default to AdamW, both with customizable parameters and optional LARS

This library is extended with multiple training techniques including the following:
- MultiAug/MultiCrop as in DINO (Caron et al., 2021: https://arxiv.org/abs/2104.14294)
- Optional ResNet modification for small images (e.g., CIFAR) common in other ResNet codes
- Optional ViT Patch Projection Freeze following MoCoV3 (Chen et al., 2021: https://arxiv.org/abs/2104.02057)
- Optimal predictor (non-learnable) following DirectSet (Wang et al., 2022: https://arxiv.org/abs/2110.04947)
- Adversarial SSL pretraining (Jiang et al., 2020: https://arxiv.org/abs/2010.13337)
- Adversarial supervised finetuning (Goodfellow et al., 2014: https://arxiv.org/abs/1412.6572 and Madry et al., 2017: https://arxiv.org/abs/1706.06083)
- Multi-task training via SSL+SL and optional adversarial training (custom code).
- SSL poison generation (He et al., 2023: https://arxiv.org/abs/2202.11202)

The user may track many important training metrics throughout training via the useProbes option.
Users are encouraged to understand what each metric measures.

This code processes models and data on CUDA - CPU-only training is not available.
Training on 1 GPU is permissible with or without DistributedDataParallel. 
Training on multiple GPUs is permissible with DistributedDataParallel

### Citation

If you use this repo, please cite the URL: https://github.com/JStyborski/VESPR
This citation will be updated once the corresponding VESPR paper is published in ECCV 2024.

### Preparation

Required libraries
- PyTorch (>v2.1.0): Installs along with torchvision and torchaudio from https://pytorch.org/get-started/locally/
- torchmetrics: Recommend installing along with PyTorch-Lightning, as in "conda install -c conda-forge pytorch-lightning"
- timm: Recommend installing along with HuggingFace, as in "conda install -c fastai timm"

Some auxiliary libraries (used by portions of the code that are currently unused or commented out) include matplotlib, torchattacks, scipy, sklearn, and probably others.

### SSL Pretraining

File: SSL_Pretrain.py

Pretrain an SSL model (including encoder, projector, predictor, and momentum encoder/projector) using a specified SSL training algorithm.
This script allows a high degree of control over encoder/projector/predictor/momentum component networks, assembled SSL architecture, and SSL loss formulations.
This script also supports self-supervised adversarial training, similar to the A2A or A2S methods found in ACL (Jiang et al., 2020: https://arxiv.org/abs/2010.13337).
It is up to the user to decide the best component networks, model architecture, and loss formulation and define settings accordingly.
It is recommended that the user review and understand the available settings within the code.
SSL_Pretrain.py supports DistributedDataParallel and command line arguments.
The default model architecture and loss is SimCLR (Chen et al., 2020: https://arxiv.org/abs/2002.05709)
Most default settings follow the CP paper (He et al., 2023: https://arxiv.org/abs/2202.11202).

Example command line inputs:

Pretrain a ResNet18 via SimCLR, MoCoV3, SimSiam, BYOL, VICReg, and DINO (respectively):
```
python SSL_Pretrain.py --trainRoot ImageNet100/train --filePrefix SimCLR_IN100
python SSL_Pretrain.py --trainRoot ImageNet100/train --filePrefix MoCo_IN100 --prdHidDim 512 --momEncBeta 0.99 --applySG True
python SSL_Pretrain.py --trainRoot ImageNet100/train --filePrefix SimSiam_IN100 --prdHidDim 512 --applySG True --winceBeta 0.0
python SSL_Pretrain.py --trainRoot ImageNet100/train --filePrefix BYOL_IN100 --prdHidDim 512 --momEncBeta 0.999 --applySG True --winceBeta 0.0
python SSL_Pretrain.py --trainRoot ImageNet100/train --filePrefix VICReg_IN100 --prjArch vicreg --symmetrizeLoss False --lossType vicreg
python SSL_Pretrain.py --trainRoot ImageNet100/train --filePrefix DINO_IN100 --prjArch dino_vit --momEncBeta 0.999 --lossType dino
```

Pretrain a ViT-Small encoder via SimCLR using LARS and A2S adversarial training and without loss symmetry on 2 GPUs:
```
CUDA_VISIBLE_DEVICES=0,1 python SSL_Pretrain.py --trainRoot ImageNet100/train --filePrefix SimCLR_IN100 --encArch vit_small --initLR 0.0005 --useLARS True --symmetrizeLoss False --useAdvList True False 
```

### Model Finetuning / Linear Probe

File: SSL_Finetune.py

Apply a linear classifier head to a pretrained encoder and train it via linear-probe (frozen encoder) or full finetuning (trainable encoder).
Besides its reliance on a pretrained encoder, this script is a straightforward supervised learning script.
This script can perform supervised learning from scratch, using the 'ftFromNull' argument.
This script also supports supervised adversarial training, with PGD settings verified to match the performance of the torchattacks library.
SSL_Finetune.py supports DistributedDataParallel and command line arguments.
Most default settings follow the CP paper (He et al., 2023: https://arxiv.org/abs/2202.11202).
The script works as follows:
- Search a specified directory for pretrained models and load one
- Create training dataset, model with linear classifier, and optimizer
- Train classifier layer (and optionally model) via supervised learning
- Save the trained model
- Repeat for remaining pretrained models

Example command line inputs:

Train linear probe on a pretrained model:
```
python SSL_Finetune.py --ptFile Trained_Models/SimCLR_IN100_pt_400.pth.tar --trainRoot ImageNet100/train --filePrefix IN100
```

Train a network via supervised learning from scratch (no pretraining):
```
python SSL_Finetune.py --trainRoot ImageNet100/train --filePrefix SL_IN100 --ftType ft --ftFromNull True --initLR 0.5 --batchSize 128 --decayLR cosdn --nEpochs 100 --weightDecay 0.0001
```

### Trained SSL or SL Model Evaluation

File: SSL_Evaluate.py

Evaluate the specified pretrained and finetuned models for KNN accuracy, train accuracy, test accuracy, and adversarial accuracy.
Pretrained models are evaluated on representation eigenvalue effective rank, representation local Lipschitz smoothness, and KNN accuracy.
Finetuned models are evaluated on train set accuracy, test set accuracy, and adversarial test accuracy.
There is an option to bypass evaluation on finetuned models if desired. This is useful if you want to evaluate a series of pretrained models.
SSL_Evaluate.py does not currently support DistributedDataParallel nor inputs via command line.

Example command line inputs:
```
python SSL_Evaluate.py
```

### SSL+SL Pretraining and Finetuning

File: SSLSL_PTFT.py

Train an encoder via combined SSL and SL losses.
This script is largely similar to SSL_Pretrain.py.

Example command line inputs:

Train a ResNet-18 encoder with contrastive loss and cross-entropy loss:
```
python SSLSL_PTFT.py --trainRoot ImageNet100/train --filePrefix SSLSL_IN100
```

Train a ResNet-18 encoder with SSL+SL but additionally with AT on the cross-entropy loss. 
This applies the nominal settings of VESPR.
```
python SSLSL_PTFT.py --trainRoot ImageNet100/train --filePrefix SSLSLAT_IN100 --useAdvList True False
```

### Trained SSL+SL Model Evaluation

File: SSLSL_Evaluate.py

Evaluate models trained via SSL+SL.
This script is largely similar to SSL_Evaluate.py.

Example command line inputs:
```
python SSLSL_Evaluate.py
```

### SSL Poison Generation

File: SSL_Poison.py

Generate SSL poisons following the "Contrastive Poisoning" (CP) method (He et al., 2023: https://arxiv.org/abs/2202.11202).
This method is similar to the "Unlearnable Examples" method (Huang et al., 2021: https://arxiv.org/abs/2101.04898).
Note that unlike SSL-based adversarial examples, CP backpropagates through augmentations to the original images.
The original CP code was implemented only for CIFAR, where all input images are small and of identical sizes.
The adapted method in this code is applicable for any image library, even those with varying image sizes, though with drawbacks to code efficiency.
SSL_Poison.py supports DistributedDataParallel and command line arguments. 
Most default settings follow the CP paper (He et al., 2023: https://arxiv.org/abs/2202.11202).
The script works as follows:
- Create a 'delta' directory of perturbation tensors that correspond to the source image directory
- At each batch load step, load the source image and corresponding perturbation tensors
- Combine source image and perturbation, then augment with multiple views
- Push augmented samples through the model and backpropagate
- Train the model or perturbations to minimize loss
- Synthesize final 'poisoned' images by saving combined source image + perturbation

Example command line inputs:

Unlearnable example poison generation for CIFAR100 on SimCLR with ResNet18 encoder:
```
python SSL_Poison.py --trainRoot CIFAR100/train --deltaRoot Poisoned_CIFAR/CP_100/deltas --poisonRoot Poisoned_CIFAR/CP_100/train --filePrefix SimCLR_CP
```

### Development Notes

In adapting the above methods, the focus was on customizeability. Not all features of all methods are exactly replicable. Caveats include the following:
- Contrastive loss is implemented via an explicit split between positive and negative samples following the method in BYOL (Grill et al., 2020: https://arxiv.org/abs/2006.07733).
  - A beta coefficient is used to weight the contrastive terms: beta=0.0 ignores contrastive terms (as in SimSiam and BYOL), beta=1.0 fully considers contrastive terms (as in SimCLR and MoCoV3).
- The projector architectures included in the code have the correct layers types, but it is up to the user to set the desired hidden, bottleneck, and output sizes.
- With multiview/multicrop (>2 views) and symmetrized loss, loss functions calculate loss between all permutations of online/target outputs for different views and average the results
  - This may be different from official implementations of MultiCrop (Caron et al., 2021: https://arxiv.org/abs/2006.09882).
- SSL adversarial training can replicate the A2A method, but cannot exactly replicate the A2S method (Jiang et al., 2020: https://arxiv.org/abs/2010.13337).
  - In A2S, Jiang et al. maintain distinct encoders/projectors with separate batchnorm stats, but this strategy is not replicated in this repository.
- I have investigated adding an accumulate_iter capability, wherein the model accumulates gradients over n batches and then updates the model on the nth batch, but have decided against implementing it
  1) accumIter assumes linear loss wrt batch size, which is not true for some SSL losses, e.g., InfoNCE 
  2) If adversarial samples are used, that process zeros model gradients anyway
  3) It makes the code messier due to multiple scalings and additional arguments passed between functions
  4) It's not necessary for most models and datasets I work with - just increase batch size
  - The implementation is not difficult - add an args.accumIter argument, scale initLR x accumIter and lossVal / accumIter, lock zero_grad() and step() functions inside args.accumIter conditional
