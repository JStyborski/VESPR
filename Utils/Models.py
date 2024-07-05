import copy
import torch
from torch import nn
import torchvision.models as models
from timm.models.vision_transformer import VisionTransformer
from timm.layers.pos_embed_sincos import build_sincos2d_pos_embed

class Base_Model(nn.Module):
    def __init__(self, encArch=None, rnCifarMod=False, vitPosEmb='normal', vitPPFreeze=True, prjArch='moco', prjHidDim=2048, prjBotDim=256,
                 prjOutDim=2048, prdHidDim=512, prdAlpha=None, prdEps=0.3, prdBeta=0.5, momEncBeta=0, applySG=True):
        super(Base_Model, self).__init__()
        self.prdAlpha = prdAlpha
        self.prdEps = prdEps
        self.prdBeta = prdBeta
        self.momZCor = None  # Initialize momentum correlation matrix as None (overwritten later)
        self.momEncBeta = momEncBeta
        self.applySG = applySG

        if 'resnet' in encArch.lower():
            # Use TorchVision ResNets
            self.encoder = models.__dict__[encArch](zero_init_residual=True)
            self.encoder.fc = nn.Identity()
            if rnCifarMod:
                # ResNet mod for CIFAR images
                self.encoder.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2, bias=False)
                self.encoder.maxpool = nn.Identity()
        elif 'vit' in encArch.lower():
            self.encoder = timm_vit(encArch)
            if vitPosEmb == 'sincos':
                posEmbWeights = build_sincos2d_pos_embed([int(self.encoder.pos_embed.shape[1] ** 0.5), int(self.encoder.pos_embed.shape[1] ** 0.5)],
                                                         int(self.encoder.pos_embed.shape[-1]))
                if self.encoder.has_class_token and not self.encoder.no_embed_class:
                    posEmbWeights = torch.cat([torch.zeros(1, self.encoder.pos_embed.shape[-1]), posEmbWeights], dim=0)
                self.encoder.pos_embed.data.copy_(posEmbWeights.unsqueeze(0))
            if vitPPFreeze:
                # Apply patch projector freeze as in https://arxiv.org/abs/2104.02057
                self.encoder.patch_embed.proj.weight.requires_grad = False
                self.encoder.patch_embed.proj.bias.requires_grad = False
                self.encoder.pos_embed.requires_grad = False

        # Get encoding dimension
        self.encDim = self.encoder.inplanes if 'resnet' in encArch else self.encoder.num_features

        self.projector = build_proj(prjArch, self.encDim, prjHidDim, prjBotDim, prjOutDim)

        if prdHidDim > 0 and self.prdAlpha is None:
            self.predictor = nn.Sequential(
                nn.Linear(prjOutDim, prdHidDim, bias=False),
                nn.BatchNorm1d(prdHidDim),
                nn.ReLU(inplace=True),
                nn.Linear(prdHidDim, prjOutDim, bias=True)
            )
        else:
            self.predictor = nn.Identity(prjOutDim)

        if self.momEncBeta > 0.0:

            # Copy the encoder, reset weights, and freeze weights
            self.momentum_encoder = copy.deepcopy(self.encoder)
            self.momentum_encoder.apply(fn=reset_model_weights)
            for param in self.momentum_encoder.parameters(): param.requires_grad = False

            # Copy the projector, reset weights, and freeze weights
            self.momentum_projector = copy.deepcopy(self.projector)
            self.momentum_projector.apply(fn=reset_model_weights)
            for param in self.momentum_projector.parameters(): param.requires_grad = False

    def forward(self, x):
        """
        Propagate input through encoder and decoder
        :param x: [tensor] [m x d] - Input tensor
        :return p: [tensor] [m x prjOutDim] - Predictor output
        :return z: [tensor] [m x prjOutDim] - Projector output
        :return r: [tensor] [m x encDim] - Encoder output
        :return mz: [tensor] [m x prjOutDim] - Momentum encoder/projector output
        """

        r = self.encoder(x)
        z = self.projector(r)

        # Run predictor or optimal predictor
        if self.prdAlpha is None:
            p = self.predictor(z)
        else:
            Wp = self.calculate_optimal_predictor(z)
            p = z @ Wp

        # Run momentum encoder or treat momEnc as regular encoder
        if self.momEncBeta > 0.0:
            mz = self.momentum_projector(self.momentum_encoder(x))
        else:
            mz = z

        # Apply stop-gradient to second branch
        if self.applySG:
            mz = mz.detach()

        return p, z, r, mz

    def calculate_optimal_predictor(self, z):
        """
        Calculate the spectral filter of z covariance to apply in place of a predictor
        :param z: [tensor] [m x d] - Input tensor, output of projector
        :return Wp: [tensor] [d x d] - Spectrally filtered, normalized, and regularized correlation matrix
        """

        # Use stop-gradient on calculating optimal weights matrix (I think)
        with torch.no_grad():

            # Calculate projector output correlation matrix
            zCor = 1 / z.size(0) * torch.tensordot(z, z, dims=([0], [0]))

            # Momentum update (or initialize) correlation matrix
            if self.momZCor is not None:
                self.momZCor = self.prdBeta * self.momZCor + (1 - self.prdBeta) * zCor
            else:
                self.momZCor = zCor

            # Calculate exponential of correlation matrix and then optimal predictor weight matrix
            # Note that I've tested multiple values of alpha. alpha=0.5 works well, alpha=1.0 causes complete collapse
            # The DirectSet paper mentions that for alpha=1.0, fAlpha or Wp needs regularization and normalization
            # I thought this referred to the prdEps term and matrix norm of fAlpha, but apparently that's not enough
            corEigvals, corEigvecs = torch.linalg.eigh(self.momZCor)
            corEigvals = torch.clamp(torch.real(corEigvals), 0)
            corEigvecs = torch.real(corEigvecs)
            fAlpha = corEigvecs @ torch.diag(torch.pow(corEigvals, self.prdAlpha)) @ torch.transpose(corEigvecs, 0, 1)
            # Shortcut: ||fAlpha||_spec = torch.linalg.matrix_norm(fAlpha, ord=2) = corEigval[-1].pow(self.prdAlpha)
            Wp = fAlpha / corEigvals[-1].pow(self.prdAlpha) + self.prdEps * torch.eye(z.size(1), device=torch.device(z.get_device()))

        return Wp

    def update_momentum_network(self):
        if self.momEncBeta > 0.0:
            for enc_params, mom_enc_params in zip(self.encoder.parameters(), self.momentum_encoder.parameters()):
                mom_enc_params.data = self.momEncBeta * mom_enc_params.data + (1 - self.momEncBeta) * enc_params.data
            for prj_params, mom_prj_params in zip(self.projector.parameters(), self.momentum_projector.parameters()):
                mom_prj_params.data = self.momEncBeta * mom_prj_params.data + (1 - self.momEncBeta) * prj_params.data


class PTFT_Model(Base_Model):
    def __init__(self, encArch=None, rnCifarMod=False, vitPosEmb='normal', vitPPFreeze=True, prjArch='moco', prjHidDim=2048, prjBotDim=256,
                 prjOutDim=2048, prdHidDim=512, prdAlpha=None, prdEps=0.3, prdBeta=0.5, momEncBeta=0, applySG=True, linClsSG=False, nClasses=None):
        super().__init__(encArch, rnCifarMod, vitPosEmb, vitPPFreeze, prjArch, prjHidDim, prjBotDim, prjOutDim, prdHidDim, prdAlpha, prdEps, prdBeta, momEncBeta, applySG)
        self.linClsSG = linClsSG
        self.nClasses = nClasses

        if self.nClasses is not None:
            self.lincls = nn.Linear(self.encDim, self.nClasses)

    def forward(self, x):
        """
        Same as Base_Model.forward, but additionally runs self.lincls and returns classifier output
        :param x: [tensor] [m x d] - Input tensor
        :return p: [tensor] [m x prjOutDim] - Predictor output
        :return z: [tensor] [m x prjOutDim] - Projector output
        :return r: [tensor] [m x encDim] - Encoder output
        :return mz: [tensor] [m x prjOutDim] - Momentum encoder/projector output
        :return c: [tensor] [m x nClasses] - Classifier output
        """

        r = self.encoder(x)
        z = self.projector(r)

        # Run predictor or optimal predictor
        if self.prdAlpha is None:
            p = self.predictor(z)
        else:
            Wp = self.calculate_optimal_predictor(z)
            p = z @ Wp

        # Run momentum encoder or treat momEnc as regular encoder
        if self.momEncBeta > 0.0:
            mz = self.momentum_projector(self.momentum_encoder(x))
        else:
            mz = z

        # Apply stop-gradient to second branch
        if self.applySG:
            mz = mz.detach()

        # Determine linear classifier output
        if self.nClasses is not None:
            if self.linClsSG:
                c = self.lincls(r.detach())
            else:
                c = self.lincls(r)
        else:
            c = None

        return p, z, r, mz, c


# Function to reinitialize all resettable weights of a given model
def reset_model_weights(layer):
    if hasattr(layer, 'reset_parameters'):
        layer.reset_parameters()
    elif hasattr(layer, 'children'):
        for child in layer.children():
            reset_model_weights(child)

# Function to define ViT architectures from the TIMM library
def timm_vit(arch, img_size=224, patch_size=16, **kwargs):
    if 'tiny' in arch.lower():
        model = VisionTransformer(img_size=img_size, patch_size=patch_size, embed_dim=192, depth=12, num_heads=3, num_classes=0, dynamic_img_size=True, weight_init='moco', **kwargs)
    elif 'small' in arch.lower():
        model = VisionTransformer(img_size=img_size, patch_size=patch_size, embed_dim=384, depth=12, num_heads=6, num_classes=0, dynamic_img_size=True, weight_init='moco', **kwargs)
    elif 'base' in arch.lower():
        model = VisionTransformer(img_size=img_size, patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, num_classes=0, dynamic_img_size=True, weight_init='moco', **kwargs)
    elif 'large' in arch.lower():
        model = VisionTransformer(img_size=img_size, patch_size=patch_size, embed_dim=1024, depth=24, num_heads=16, num_classes=0, dynamic_img_size=True, weight_init='moco', **kwargs)
    return model


class L2_Norm_Layer(nn.Module):

    def __init__(self, dim=1, eps=1e-12):
        super(L2_Norm_Layer, self).__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x):
        return nn.functional.normalize(x, p=2, dim=self.dim, eps=self.eps)


def build_proj(arch, encDim, prjHidDim, prjBotDim, prjOutDim):

    if arch == 'simsiam' or arch == 'simclr' or arch == 'mec':
        proj = nn.Sequential(
            nn.Linear(encDim, encDim, bias=False),
            nn.BatchNorm1d(encDim),
            nn.ReLU(inplace=True),
            nn.Linear(encDim, encDim, bias=False),
            nn.BatchNorm1d(encDim),
            nn.ReLU(inplace=True),
            nn.Linear(encDim, prjOutDim, bias=False),
            nn.BatchNorm1d(prjOutDim, affine=False)
        )

    if arch == 'moco':
        proj = nn.Sequential(
            nn.Linear(encDim, prjHidDim, bias=False),
            nn.BatchNorm1d(prjHidDim),
            nn.ReLU(inplace=True),
            nn.Linear(prjHidDim, prjHidDim, bias=False),
            nn.BatchNorm1d(prjHidDim),
            nn.ReLU(inplace=True),
            nn.Linear(prjHidDim, prjOutDim, bias=False),
            nn.BatchNorm1d(prjOutDim, affine=False)
        )

    elif arch == 'byol':
        proj = nn.Sequential(
            nn.Linear(encDim, prjHidDim, bias=False),
            nn.BatchNorm1d(prjHidDim),
            nn.ReLU(inplace=True),
            nn.Linear(prjHidDim, prjOutDim, bias=False),
        )

    elif arch == 'bt' or arch == 'vicreg':
        proj = nn.Sequential(
            nn.Linear(encDim, prjHidDim, bias=False),
            nn.BatchNorm1d(prjHidDim),
            nn.ReLU(inplace=True),
            nn.Linear(prjHidDim, prjHidDim, bias=False),
            nn.BatchNorm1d(prjHidDim),
            nn.ReLU(inplace=True),
            nn.Linear(prjHidDim, prjOutDim, bias=False),
        )

    elif arch == 'dino_cnn':
        proj = nn.Sequential(
            nn.Linear(encDim, prjHidDim, bias=True),
            nn.BatchNorm1d(prjHidDim),
            nn.GELU(),
            nn.Linear(prjHidDim, prjHidDim, bias=True),
            nn.BatchNorm1d(prjHidDim),
            nn.GELU(),
            nn.Linear(prjHidDim, prjBotDim, bias=True),
            L2_Norm_Layer(dim=1),
            nn.utils.parametrizations.weight_norm(nn.Linear(prjBotDim, prjOutDim, bias=False))
        )
        proj[-1].parametrizations.weight.original0.data.fill_(1.)
        proj[-1].parametrizations.weight.original0.requires_grad = False

    elif arch == 'dino_vit':
        proj = nn.Sequential(
            nn.Linear(encDim, prjHidDim, bias=True),
            nn.GELU(),
            nn.Linear(prjHidDim, prjHidDim, bias=True),
            nn.GELU(),
            nn.Linear(prjHidDim, prjBotDim, bias=True),
            L2_Norm_Layer(dim=1),
            nn.utils.parametrizations.weight_norm(nn.Linear(prjBotDim, prjOutDim, bias=False))
        )
        proj[-1].parametrizations.weight.original0.data.fill_(1.)
        proj[-1].parametrizations.weight.original0.requires_grad = False

    return proj
