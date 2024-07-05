import numpy as np
import PIL

import torch

def scale_vecs(xArr, p, eps):
    """
    Scale each vector of an array based their corresponding norms such that scaled vector norms do not exceed eps
    :param xArr: [array] [m x n] - Array of multiple vectors
    :param p: [float] [1] - Norm order
    :param eps: [float] [1] - Maximum norm for each vector in an array
    :return scaled: [array] [m x n]
    """
    norm = np.linalg.norm(xArr, axis=1, p=p, keepdims=True)
    scaled = xArr * np.clip(eps / (norm + 1e-6), a_max=1.)  # Do not upscale inputs with norms lower than eps
    return scaled


def clip_vecs(xArr, p, eps):
    """
    For each vector in array, ||vec||_inf>eps, project the vector to nearest point such that ||vec||_inf=eps
    This process is identical to clipping any vector elements that are above/below the threshold
    For norms other than infinity, clipping is treated the same as scaling
    :param xArr: [array] [m x n] - Array of multiple vectors
    :param p: [float] [1] - Norm order
    :param eps: [float] [1] - Maximum norm for each vector in an array
    :return clipped: [array] [m x n]
    """
    if p == np.inf:
        clipped = np.clip(xArr, a_min=-eps, a_max=eps)
    else:
        clipped = scale_vecs(xArr, p, eps)
    return clipped


def cross_cov(xArr1, xArr2):
    """
    Calculate the cross-covariance between two arrays of vectors
    inp xArr1 [array or tensor] [m x n] - Array of multiple vectors
    inp xArr2 [array or tensor] [m x n] - Array of multiple vectors
    ret crossCov [array] [n x n]
    """
    xMeanArr1 = np.mean(xArr1, axis=0, keepdims=True)
    xMeanArr2 = np.mean(xArr2, axis=0, keepdims=True)
    crossCov = 1 / xArr1.shape[0] * (np.transpose(xArr1 - xMeanArr1) @ (xArr2 - xMeanArr2))
    return crossCov


def cross_corr(xArr1, xArr2):
    """
    Calculate the cross-correlation between two arrays of vectors
    :param xArr1: [array or tensor] [m x n] - Array of multiple vectors
    :param xArr2: [array or tensor] [m x n] - Array of multiple vectors
    :return crossCor: [array] [n x n]
    """
    crossCor = 1 / xArr1.shape[0] * (np.transpose(xArr1) @ xArr2)
    return crossCor


def dist_bt_vecs(xArr1, xArr2, p=2):
    """
    Calculate the distance between corresponding vectors in arrays
    :param xArr1: [array or tensor] [m x n] - Array of multiple vectors
    :param xArr2: [array or tensor] [m x n] - Array of multiple vectors
    :param p: [float] [1, 2, or np.inf] - Norm to use for distance calculation
    :return distArr: [array] [m x 1]
    """
    distArr = np.linalg.norm(xArr1 - xArr2, ord=p, axis=1, keepdims=True)
    return distArr


def dist_within_array(xArr, p=2):
    """
    Calculate the distance between every vector in an array with every other vector in the same array
    :param xArr: [array or tensor] [m x n] - Array of multiple vectors
    :param p: [float] [1, 2, or np.inf] - Norm to use for distance calculation
    :return distArr: [array] [m * (m-1) / 2]
    """
    distList = []
    for i in range(xArr.shape[0] - 1):
        for j in range(i + 1, xArr.shape[0]):
            distList.append(dist_bt_vecs(xArr[[i], :], xArr[[j], :], p=p)[0, 0])
    return np.array(distList)


def dist_bt_arrays(xArr1, xArr2, p=2):
    """
    Calculate the distance between every vector in one array and every vector in another array
    :param xArr1: [array or tensor] [m x n] - Array of multiple vectors
    :param xArr2: [array or tensor] [m x n] - Array of multiple vectors
    :param p: [float] [1, 2, or np.inf] - Norm to use for distance calculation
    :return distList: [array] [m * n]
    """
    distList = []
    for i in range(xArr1.shape[0]):
        for j in range(xArr2.shape[0]):
            distList.append(dist_bt_vecs(xArr1[[i], :], xArr2[[j], :], p=p)[0, 0])
    return np.array(distList)


def cos_sim_bt_vecs(xArr1, xArr2, normalize=True):
    """
    Calculate the cosine similarity between corresponding vectors in arrays
    :param xArr1: [array or tensor] [b x d] - Array of multiple vectors
    :param xArr2: [array or tensor] [b x d] - Array of multiple vectors
    :param normalize: [Boolean] [1] - Whether to divide the dot product by the vector magnitudes
    :return cosSimArr: [array] [b x 1]
    """
    if normalize:
        xArr1 /= np.linalg.norm(xArr1, axis=1, keepdims=True)
        xArr2 /= np.linalg.norm(xArr2, axis=1, keepdims=True)
    cosSimArr = np.sum(xArr1 * xArr2, axis=1, keepdims=True)
    return cosSimArr


def cos_sim_within_array(xArr, normalize=True):
    """
    Calculate the cosine similarity between every vector in an array with every other vector in the same array
    :param xArr: [array or tensor] [b x d] - Array of multiple vectors
    :param normalize: [Boolean] [1] - Whether to divide the dot product by the vector magnitudes
    :return simList: [list of floats] [1]
    """
    if normalize:
        xArr /= np.linalg.norm(xArr, axis=1, keepdims=True)
    cosSimArr = np.tensordot(xArr, xArr, axes=(-1, -1))
    cosSimArr = cosSimArr[np.triu_indices(m=cosSimArr.shape[0], k=1, n=cosSimArr.shape[1])].reshape(1, -1)[0]
    return cosSimArr


def cos_sim_bt_arrays(xArr1, xArr2, normalize=True):
    """
    Calculate the cosine similarity between every vector in one array and every vector in another array
    :param xArr1: [array or tensor] [b x d] - Array of multiple point vectors
    :param xArr2: [array or tensor] [b x d] - Array of multiple point vectors
    :param normalize: [Boolean] [1] - Whether to divide the dot product by the vector magnitudes
    :return simList: [list of floats] [1]
    """
    if normalize:
        xArr1 /= np.linalg.norm(xArr1, axis=1, keepdims=True)
        xArr2 /= np.linalg.norm(xArr2, axis=1, keepdims=True)
    cosSimArr = np.tensordot(xArr1, xArr2, axes=(-1, -1)).reshape(1, -1)[0]
    return cosSimArr


def softmax(xArr):
    """
    Calculate the softmax vectors for each vector in an array
    :param xArr: [array or tensor] [b x d] - Array of multiple point vectors
    :return softArr: [array or tensor] [b x d] - Softmax version of array
    """
    xArr = xArr - np.max(xArr, axis=1, keepdims=True) # Shift away from large values to prevent overflow
    softArr = np.exp(xArr) / np.sum(np.exp(xArr), axis=1, keepdims=True)
    return softArr


def log_softmax(xArr):
    """
    Numerically stable approximation to full log softmax
    log(exp(x)/∑(exp(x))) = x − log(∑(exp(x))) ~ x - max(x)
    :param xArr:
    :return:
    """
    logSoftArr = xArr - np.max(xArr, axis=1, keepdims=True)
    return logSoftArr


def kl_div_bt_vecs(xArr1, xArr2, use_softmax=True):
    """
    Calculate the KL divergence between corresponding vectors in arrays
    :param xArr1: [array or tensor] [b x d] - Array of multiple vectors
    :param xArr2: [array or tensor] [b x d] - Array of multiple vectors
    :param use_softmax: [Boolean] [1] - Whether to apply softmax to each array first
    :return klDivArr: [array] [b x 1]
    """
    if use_softmax:
        xArr1 = softmax(xArr1)
        xArr2 = softmax(xArr2)
    klDivArr = -1.0 * np.sum(xArr1 * np.log(np.clip(xArr2 / xArr1, np.finfo(np.float32).eps, None)), axis=1, keepdims=True)
    return klDivArr


def cross_ent_bt_vecs(xArr1, xArr2, use_softmax=True):
    """
    Calculate the cross-entropy between corresponding vectors in arrays
    :param xArr1: [array or tensor] [b x d] - Array of multiple vectors
    :param xArr2: [array or tensor] [b x d] - Array of multiple vectors
    :param use_softmax: [Boolean] [1] - Whether to apply softmax to each array first
    :return crossEntArr: [array] [b x 1]
    """
    if use_softmax:
        xArr1 = softmax(xArr1)
        #xArr2 = log_softmax(xArr2)
        xArr2 = softmax(xArr2)
    #crossEntArr = -1.0 * np.sum(xArr1 * xArr2, axis=1, keepdims=True)
    crossEntArr = -1.0 * np.sum(xArr1 * np.log(np.clip(xArr2, np.finfo(np.float32).eps, None)), axis=1, keepdims=True)
    return crossEntArr


def cross_ent_within_array(xArr):
    """
    Calculate the cross-entropy between every vector in an array with every other vector in the same array
    :param xArr: [array or tensor] [b x d] - Array of multiple vectors
    :return ceList: [list of floats] [1]
    """
    ceList = []
    for i in range(xArr.shape[0] - 1):
        for j in range(i + 1, xArr.shape[0]):
            ceList.append(cross_ent_bt_vecs(xArr[[i], :], xArr[[j], :])[0, 0])
    return ceList


def cross_ent_bt_arrays(xArr1, xArr2):
    """
    Calculate the cross-entropy between every vector in one array and every vector in another array
    :param xArr1: [array or tensor] [b x d] - Array of multiple point vectors
    :param xArr2: [array or tensor] [b x d] - Array of multiple point vectors
    :return simList: [list of floats]
    """
    ceList = []
    for i in range(xArr1.shape[0]):
        for j in range(xArr2.shape[0]):
            ceList.append(cross_ent_bt_vecs(xArr1[[i], :], xArr2[[j], :])[0, 0])
    return ceList


def infonce_bound(xArr1, xArr2, normalize=True, temperature=1.0):
    """
    Calculate the InfoNCE bound for mutual information between views: I(v1;v2) ≥ log(2b-1) - L_NCE_i = I_NCE(v1;v2)
    Equation obtained from InfoMin paper: https://arxiv.org/abs/2005.10243
    Implementation of InfoNCE loss from SimCLR paper: https://arxiv.org/abs/2002.05709
    :param xArr1: [array] [b x d] - Array of multiple vectors
    :param xArr2: [array] [b x d] - Array of multiple vectors
    :param normalize: [Boolean] - Whether to divide the dot product by the vector magnitudes
    :param temperature: [float] - Temperature factor applied to similarity terms
    :return bound: [float]
    """

    # Calculate loss term due to positive similarities
    posLoss = -1.0 * np.mean(cos_sim_bt_vecs(xArr1, xArr2, normalize))

    # Calculate loss term due to negative similarities (both within batch and between batches)
    if normalize:
        xArr1 /= np.linalg.norm(xArr2, axis=1, keepdims=True)
        xArr2 /= np.linalg.norm(xArr2, axis=1, keepdims=True)
    negSameSim = np.tensordot(xArr1, xArr1, axes=(-1, -1))
    np.fill_diagonal(negSameSim, 0.0)
    negDiffSim = np.tensordot(xArr1, xArr2, axes=(-1, -1))
    negLoss = np.mean(np.log(np.sum(np.exp(negSameSim / temperature), axis=-1, keepdims=True) +
                               np.sum(np.exp(negDiffSim / temperature), axis=-1, keepdims=True)))

    # Calculate the final bound
    bound = np.log(2 * xArr1.shape[0] - 1) - (posLoss / temperature + negLoss)

    return bound


def array_sparsity(xArr):
    """
    Count the number of nonzero elements in every vector in an array
    :param xArr: [array or tensor] [m x n] - Array of multiple vectors
    :return sparseArr: [array of ints] [m x 1]
    """
    nonZeroBool = np.abs(xArr) >= np.finfo(float).eps
    sparseArr = np.sum(nonZeroBool, axis=1, keepdims=True)
    return sparseArr


def array_near_zero(xArr):
    """
    Count the number of elements within 1 OoM of underflow in every vector in an array
    :param xArr: [array or tensor] [m x n] - Array of multiple vectors
    :return nearZeroList: [list of ints] [m x 1]
    """
    nearZeroBool = np.logical_and(np.abs(xArr) >= np.finfo(np.float32).eps,
                                  np.abs(xArr) <= 10 * np.finfo(np.float32).eps)
    nearZeroArr = np.sum(nearZeroBool, axis=1, keepdims=True)
    return nearZeroArr


def array_eigdecomp(xArr, covOrCor='cov', sorted=True, forceReal=True):
    """
    Get the (possibly) sorted (desc order) eigenvalues and eigenvectors of the covariance of an array of vectors
    :param xArr: [array] [b x d] - Array of multiple vectors
    :param covOrCor: [string] - 'cov' or 'cor' to indicate whether to measure cross-correlation or covariance
    :param sorted: [Boolean] - Boolean value to sort eigenvalues in descending order
    :param forceReal: [Boolean] - Boolean value to force real outputs
    :return eigval: [array] [d]
    :return eigvec: [array] [d x d]
    """
    if covOrCor == 'cov':
        coArr = cross_cov(xArr, xArr)
    elif covOrCor == 'cor':
        coArr = cross_corr(xArr, xArr)
    eigval, eigvec = np.linalg.eigh(coArr)
    if forceReal:
        eigval = np.real(eigval)
    if sorted:
        sortIdx = np.argsort(eigval)[::-1]
        eigval = eigval[sortIdx]
        eigvec = eigvec[:, sortIdx]
    return eigval, eigvec


def array_to_reduced_eigspace(xArr, redDim=None):
    """
    Project an array of vectors to a (possibly) reduced eigenspace of its covariance matrix
    :param xArr: [array or tensor] [m x n] - Array of multiple vectors
    :param redDim: [int] [1] - Number of reduced dimensions for eigenspace
    :return reducedEigSpace: [array] [redDim x n]
    :return eigval: [array] [n]
    :return eigvec: [array] [n x n]
    """
    if redDim is None:
        redDim = xArr.shape[1]
    eigval, eigvec = array_eigdecomp(xArr)
    reducedEigSpace = np.dot(xArr, eigvec[:, :redDim])
    return reducedEigSpace, eigval, eigvec


def eigspace_reduce_reconstruct(xArr, redDim=None):
    """
    Project an array of vectors to a reduced eigenspace and then back to the original space
    :param xArr: [array or tensor] [m x n] - Array of multiple vectors
    :param redDim: [int] [1] - Number of reduced dimensions for eigenspace
    :return reconstructedEncArr: [array] [m x n]
    :return eigval: [array] [n]
    :return eigvec: [array] [n x n]
    """
    reducedEigSpace, eigval, eigvec = array_to_reduced_eigspace(xArr, redDim)
    reconstructedEncArr = np.dot(reducedEigSpace, eigvec[:, :redDim].T)
    return reconstructedEncArr, eigval, eigvec


def spectral_filter(z, power=0.0, cutoff=None):
    # For z (n x d) and Cz (d x d), z = U Sigz V.T, Cz = 1/n z.T z = Q Lamz Q.T, with Q = V and Lamz = 1/n Sigz^2
    # Spectral filter g(Lam) adjusts eigenvalues and then applies W = V g(Lamz) V.T on z, p = z @ W
    # This affects output correlation: Cp = V g(Lamz)^2 Lam V.T, such that Lamp = g(Lamz)^2 Lamz
    # Low pass filter emphasizes large eigvals and diminishes low eigvals - high pass filter vice versa
    # In this function we specifically apply g(Lamz) = Lamz.pow(power)
    # power should be between -0.5 and +1.0 - [-0.5, 0] gives high pass filter, [0, 1.0] gives low pass filter
    # Power examples: -0.5 -> Lamp = I, 0 -> Lamp = Lamz, 0.5 -> Lamp = Lamz^2, 1.0 -> Lamp = Lamz^3
    U, Sigz, VT = np.linalg.svd(z, full_matrices=False)
    Lamz = 1 / z.size(0) * Sigz.clip(0).pow(2)
    Lamp = Lamz
    if power is not None:
        Lamp = Lamz.pow(1 + 2 * power)
    if cutoff is not None:
        Lamp[cutoff:] = 0
    Sigp = Lamp.sqrt() * z.size(0) ** 0.5
    specZ = U @ np.diag(Sigp) @ VT
    return specZ


def entropy(vals):
    # Note that this formulation of using non-negative values tacitly enforces the rule that 0*log0 = 0 due to the sum
    nonzero_mask = vals > np.finfo(np.float32).eps
    nonzero_vals = vals[nonzero_mask]
    entropy = -1.0 * np.sum(nonzero_vals * np.log(nonzero_vals))
    return entropy


def area_under_curve(vals, dx):
    """
    From a list of values, get the area under the curve of the cumulative sum of the values using Riemann sum
    :param vals: [array or list] [n] - Array or list of values
    :param dx: [array or list] [n] - Width of each element in vals, should sum to 1
    - Note Li et al. 2022 AUC uses 1/d (d=len(cumulVal)) factor but with dx of 1's
    :return auc: [float] [1]
    """
    assert abs(sum(dx) - 1) < 0.0001
    cumulVals = np.cumsum(vals)
    auc = sum(cumulVals * dx) / cumulVals[-1]
    return auc


def jacobian(xInpArr1, xInpArr2, xOutArr1, xOutArr2):
    """
    Calculate the Jacobian matrix across all samples in the input/output arrays
    Note that for m > 1, this calculates an average Jacobian across all samples
    :param xInpArr1: [array or tensor] [m x n1] - 1st input array of multiple vectors
    :param xInpArr2: [array or tensor] [m x n1] - 2nd input array of multiple vectors
    :param xOutArr1: [array or tensor] [m x n2] - Function output array of 1st input array
    :param xOutArr2: [array or tensor] [m x n2] - Function output array of 2nd input array
    :return jac: [array] [n2 x n1]
    """
    invInpDiff = 1. / (xInpArr2 - xInpArr1)
    jac = 1 / xInpArr1.shape[0] * np.transpose(xOutArr2 - xOutArr1) @ invInpDiff
    return jac


def local_lip(xInpArr1, xInpArr2, xOutArr1, xOutArr2, pIn=np.Inf, pOut=1):
    """
    Calculate the local Lipschitz value for each sample of the input/output arrays
    - 2020 Yang (https://arxiv.org/abs/2003.02460) uses pIn = inf, pOut = 1
    - 2022 Tian (https://arxiv.org/abs/2206.01342) uses pIn = pOut = 2 for "local roughness"
    :param xInpArr1: [array or tensor] [m x n1] - 1st input array of multiple vectors
    :param xInpArr2: [array or tensor] [m x n1] - 2nd input array of multiple vectors
    :param xOutArr1: [array or tensor] [m x n2] - Function output array of 1st input array
    :param xOutArr2: [array or tensor] [m x n2] - Function output array of 2nd input array
    :param pIn: Norm to use for distance calculation for numerator
    :param pOut: Norm to use for distance calculation for denominator
    :return reduced: [array] [m x 1]
    """
    lolip = dist_bt_vecs(xOutArr1, xOutArr2, p=pOut) / (dist_bt_vecs(xInpArr1, xInpArr2, p=pIn) + 1e-6)
    return lolip


def matrix_norm(xMat, type='entry', p=2):
    """
    Calculate the norm of a matrix
    - Wikipedia (https://en.wikipedia.org/wiki/Matrix_norm)
    :param xMat: [array or tensor] [n1 x n2] - Input matrix for which to calculate norm
    :param type: [string] [1] - Flag to denote norm class
    :param p: [float or string] [1] - Flag to denote norm order/type
    :return matNorm: [float] [1] - Matrix norm value
    """
    if type == 'entry':
        # Entry-wise matrix norms
        # matNorm = sum_i,j(|xMat_i,j| ^ p) ^ (1/p)
        # Note that entrywise norm with p=2 is the Frobenius norm
        if p == np.inf:
            matNorm = np.max(np.abs(xMat))
        else:
            matNorm = np.sum(np.abs(xMat) ** p) ** (1 / p)
    if type == 'ind':
        # Induced matrix norms
        if p == 1:
            # Induced 1 norm is maximum absolute column sum
            matNorm = np.max(np.sum(np.abs(xMat), axis=0))
        elif p == np.inf:
            # Induced inf norm is maximum absolute row sum
            matNorm = np.max(np.sum(np.abs(xMat), axis=1))
        elif p == 2:
            # Induced 2 norm is equivalent to spectral norm
            type = 'svd'
            p = 'spec'
    if type == 'svd':
        # SVD-based matrix norms
        _, s, _ = np.linalg.svd(xMat)
        if p == 'spec':
            # Spectral norm is maximum singular value
            matNorm = np.max(s)
        elif p == 'nuc':
            # Nuclear norm is sum of singular values
            matNorm = np.sum(s)

    return matNorm


def optimize_linear(xArr, p, eps=1):
    """
    Solves the linear optimization problem, eta = argmax(inner(input, eta)) where ||eta||_p <= eps
    This is closely related to the dual norm: ||input|| = max(inner(input, eta)) where ||eta||_p <= 1
    - Wikipedia (https://en.wikipedia.org/wiki/Dual_norm)
    - Also check FGSM_PGD.py in the Adversarial repo
    - Also check Convex Optimization notes for NTU lecture 4, slide 25
    :param xArr: [array] [m x n] - Input array to calculate dual norm
    :param p: [float] [1] - Norm order for eta
    :param eps: [float] [1] - Constraint value for norm on eta
    :return dualNorm: [float] [1]
    """
    if p == 1:
        # Linear optimal for 1-norm constrained eta is the max magnitude input vector element in the corresponding dim
        absInp = np.abs(xArr)
        maxAbsInp = np.amax(absInp, axis=1, keepdims=True)
        maxMask = absInp == maxAbsInp
        numTies = np.sum(maxMask, axis=1, keepdims=True)  # Heuristic to address max mag ties at multiple elements
        dualNorm = np.sign(xArr) * maxMask / numTies * eps
    elif p == 2:
        # Linear optimal for 2-norm constrained eta is the input vector scaled by the Euclidean norm of the input
        dualNorm = xArr / np.linalg.norm(xArr, axis=1, keepdims=True, p=2) * eps
    elif p == np.inf:
        # Linear optimal for inf-norm constrained eta is the max magnitude input vector element in every dimension
        dualNorm = np.sign(xArr) * eps
    return dualNorm


def get_knn(xArr, xVec, k, p=2):
    """
    Calculate the distance from a single vector to every vector in an array, then find the k closest vectors
    :param xArr: [array] [m x n] - Array of sample vectors
    :param xVec: [array] [1 x n] - Central point vector
    :param k: [int] [1] - Number of nearest neighbors
    :param p: [float] [1] - Norm to use for distance calculation
    :return kIdx: [list of ints] [k] - List of indices for xArr vectors that xVec is closest to
    :return kDist: [list of floats] [k] - List of distances corresponding to kIdx
    """
    distArr = np.linalg.norm(xArr - xVec, ord=p, axis=1)  # Note that this relies on broadcasting
    kIdx = np.argsort(distArr)[:k].tolist()
    kDist = [distArr[idx] for idx in kIdx]
    return kIdx, kDist


def calc_hypervolume(r, d, p):
    """
    Calculate the volume of a hyperstructure (rhomboid, sphere, cube)
    :param r: [float] [1] - Characteristic distance measure in hyperspace (see comments below)
    :param d: [float] [1] - Dimensionality of the hyperstructure
    :param p: [float] [1] - Distance measure order that determines hyperstructure type
    :return vol: [float] [1]
    """
    if p == 1:
        # Volume of a hyperrhomboid with distance from center to one corner r
        vol = (1.414214 * r) ** d
    elif p == 2:
        # Volume of a hypersphere with radius r
        vol = 3.141593 ** (d / 2) / np.math.factorial(d / 2) * r ** d
    elif p == np.inf:
        # Volume of a hypercube with distance side length 2 * r
        vol = (2 * r) ** d
    return vol


def approximate_px(xArr, xVec, k, p):
    """
    Approximate local probability using KNN to determine local density
    - Wang 2009 (https://www.princeton.edu/~kulkarni/Papers/Journals/j068_2009_WangKulVer_TransIT.pdf)
    :param xArr: [array] [m x n] - Array of sample vectors
    :param xVec: [array] [1 x n] - Central point vector
    :param k: [int] [1] - Number of nearest neighbors to use for approximating px
    :param p: [float] [1] - Distance measure order that determines hyperstructure volume type
    :return px: [float] [1] - Approximate probability at xVec
    """
    _, kDist = get_knn(xArr, xVec, k, p)
    vol = calc_hypervolume(kDist[-1], xVec.shape[0], p)
    px = k / (xArr.shape[0] * vol)
    return px


def approximate_kl(xArr1, xArr2, k, p):
    """
    Approximate the KL divergence between two arrays of points using KNN
    - Wang 2009 (https://www.princeton.edu/~kulkarni/Papers/Journals/j068_2009_WangKulVer_TransIT.pdf)
    :param xArr1: [array] [b x d] - Array of sample vectors
    :param xArr2: [array] [b x d] - Array of sample vectors
    :param k: [int] [1] - Number of nearest neighbors to use for approximating KL
    :param p: [float] [1] - Distance measure order that determines hyperstructure volume type
    :return kl: [float] [1] - Approximate KL divergence between arrays
    """
    n = xArr1.shape[0]
    sumList = []
    for i in range(n):
        xVec = xArr1[i, :]
        xArr1Remain = np.delete(xArr1, i, axis=0)
        _, pkDist = get_knn(xArr1Remain, xVec, k, p)  # Distances from xVec to topk closest vectors in xArr1
        _, qkDist = get_knn(xArr2, xVec, k, p)  # Distances from xVec to topk closest vectors in xArr2
        sumList.append(np.log(qkDist[-1] / pkDist[-1]))  # Append the log of the ratio of the largest distances
    kl = np.log(xArr2.shape[0] / (n - 1)) + xArr1.shape[1] / n * sum(sumList)
    return kl


def approximate_ce(xArr1, xArr2, k, p):
    """
    Approximate the cross-entropy between two arrays of points using KNN
    :param xArr1: [array] [m x n] - Array of sample vectors
    :param xArr2: [array] [m x n] - Array of sample vectors
    :param k: [int] [1] - Number of nearest neighbors to use for approximating KL
    :param p: [float] [1] - Distance measure order that determines hyperstructure volume type
    :return ce: [float] [1] - Approximate cross-entropy between arrays
    """
    m = xArr1.shape[0]
    sumList = []
    for i in range(m):
        xVec = xArr1[i, :]
        qx = approximate_px(xArr2, xVec, k, p)
        sumList.append(np.log(qx))
    ce = -1 / m * sum(sumList)
    return ce


def apply_kernel(inpList, kernType='gauss', kernParam=1.0):
    """
    Given a list of distances/similarities, return the corresponding list of kernelized distances/similarities
    :param inpList: [list of floats] [n] - List of distances or similarities (not normalized)
    :param kernType: [string] [1] - Flag for kernel type
    :param kernParam: [float] [1] - Custom parameter value to use for different kernels
    - sigma for Laplace/Gauss, degree for polynomial
    :return kernVals: [list of floats] [n]
    """
    # Note that Laplacian/Gaussian kernels use distances, linear/polynomial kernels use similarities
    if kernType == 'laplace':
        kernVals = [np.exp(-1. * inp / kernParam) for inp in inpList]
    elif kernType == 'gauss':
        kernVals = [np.exp(-1. * inp ** 2 / (2 * kernParam ** 2)) for inp in inpList]
    elif kernType == 'linear':
        kernVals = inpList
    elif kernType == 'polynom':
        kernVals = [(inp + 1) ** kernParam for inp in inpList]
    return kernVals


def calc_mmd(xArr1, xArr2, p=2, kernType='gauss', kernParam=1.0):
    """
    Calculate the Maximum Mean Discrepancy between two arrays of samples
    - Gretton 2012 (https://www.jmlr.org/papers/volume13/gretton12a/gretton12a.pdf)
    - Luo 2023 (https://arxiv.org/abs/2303.01289)
    :param xArr1: [array] [m x n] - Array of sample vectors
    :param xArr2: [array] [m x n] - Array of sample vectors
    :param p: [float] [1] - Distance measure order
    :param kernType: [string] [1] - Flag for kernel type
    :param kernParam: [float] [1] - Custom parameter value to use for different kernels
    :return MMD: [float] [1]
    """
    m1 = xArr1.shape[0]
    m2 = xArr2.shape[0]
    term1Dists = dist_within_array(xArr1, p=p)
    term1 = 1 / (m1 * (m1 - 1)) * sum(apply_kernel(term1Dists, kernType=kernType, kernParam=kernParam))
    term2Dists = dist_within_array(xArr2, p=p)
    term2 = 1 / (m2 * (m2 - 1)) * sum(apply_kernel(term2Dists, kernType=kernType, kernParam=kernParam))
    term3Dists = dist_bt_arrays(xArr1, xArr2, p=p)
    term3 = 2 / (m1 * m2) * sum(apply_kernel(term3Dists, kernType=kernType, kernParam=kernParam))
    mmd = np.sqrt(term1 + term2 - term3)
    return mmd


def calc_mcd(xArr1, xArr2, p=np.inf):
    """
    Calculate the Minimum Classwise Distances between samples of 1 class and all other samples
    - 2023 Luo (https://arxiv.org/abs/2303.01289)
    - 2020 Yang (https://arxiv.org/abs/2003.02460)
    :param xArr1: [array] [m x n1] - Array of sample vectors from 1 class
    :param xArr2: [array] [m x n2] - Array of sample vectors from all other classes
    :param p: [float] [1] - Distance measure order
    :return mcdList: [list of floats] [n1]
    """
    mcdList = []
    for i in range(xArr1.shape[0]):
        _, kDist = get_knn(xArr2, xArr1[i, :], k=1, p=p)
        mcdList.append(kDist[0])
    return mcdList


def calc_uniformity(xArr1, xArr2, kernParam=1.0):
    """
    Calculate the uniformity metric of a sample distribution
    - 2022 Wang (https://arxiv.org/abs/2005.10242)
    The uniformity equation assumes that samples have all been drawn IID from the population
    Therefore this code uses two separate arrays as input, assuming that they are not correlated to each other
    :param xArr1: [array] [m x n] - Array of sample vectors from population
    :param xArr2: [array] [m x n] - Array of sample vectors from population
    :param kernParam: [float] [1] - Gaussian kernel standard deviation
    :return uniformity: [float] [1]
    """
    kernDists = apply_kernel(xArr2 - xArr1, kernType='gauss', kernParam=kernParam)
    uniformity = np.log(1. / xArr1.shape[0] * sum(kernDists))
    return uniformity


def calc_mag(xInpArr1, xInpArr2, xOutArr1, xOutArr2):
    """
    Calculate Maximum Absolute Gradient for a set of vectors
    Note that for m > 1, this calculates an average Jacobian across all samples
    :param xInpArr1: [array or tensor] [m x n1] - 1st input array of multiple vectors
    :param xInpArr2: [array or tensor] [m x n1] - 2nd input array of multiple vectors
    :param xOutArr1: [array or tensor] [m x n2] - Function output array of 1st input array
    :param xOutArr2: [array or tensor] [m x n2] - Function output array of 2nd input array
    :param p: [float] [1] - Matrix norm order
    :return mag: [float] [1]
    """
    jac = jacobian(xInpArr1, xInpArr2, xOutArr1, xOutArr2)
    mag = matrix_norm(jac, type='entry', p=np.inf)
    return mag


##################
# FDA Projection #
##################

# Calculate transformation to the line that maximizes separation of projections for 2 distributions (assumes encoding_dim x samples)
def fisher_W(xArr1, xArr2):
    """
    :param xArr1: [array or tensor] [m_samples x encoding_dim] - Array of multiple point vectors
    :param xArr2: [array or tensor] [m_samples x encoding_dim] - Array of multiple point vectors
    :return W: [array] [???]
    - linear transformation matrix to project encodings to the line that maximally separate 2 distributions
    """
    xMeanArr1 = np.mean(xArr1, axis=0, keepdims=True)
    xMeanArr2 = np.mean(xArr2, axis=0, keepdims=True)
    scatter1 = np.dot(np.transpose(xArr1 - xMeanArr1), xArr1 - xMeanArr1)
    scatter2 = np.dot(np.transpose(xArr2 - xMeanArr2), xArr2 - xMeanArr2)
    totalScatter = scatter1 + scatter2
    W = np.dot(np.linalg.inv(totalScatter), xMeanArr1 - xMeanArr2)
    W = W / np.linalg.norm(W)
    return W



