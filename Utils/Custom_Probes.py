import csv
import numpy as np
import matplotlib.pyplot as plt

import Utils.Analysis_Utils as AU


class Probe:
    def __init__(self):
        self.storeList = []
        self.sum = 0.0
        self.count = 0
        self.avg = 0.0

    def store(self, x):
        self.storeList.append(x)

    def runSum(self, x):
        self.sum += x

    def runAvg(self, x):
        self.sum += x
        self.count += 1
        self.avg = self.sum / self.count


class Pretrain_Probes:

    def __init__(self):

        self.epochProbe = Probe()
        self.lossProbe = Probe()
        self.loss1Probe = Probe()
        self.loss2Probe = Probe()
        self.r1r2AugSimProbe = Probe()
        self.r1AugSimProbe = Probe()
        self.p1mz2AugSimProbe = Probe()
        self.p1AugSimProbe = Probe()
        self.r1r2AugConcProbe = Probe()
        self.r1AugConcProbe = Probe()
        self.r1VarProbe = Probe()
        self.r1CorrStrProbe = Probe()
        self.r1r2InfoBoundProbe = Probe()
        self.r1EigProbe = Probe()
        self.r1EigERankProbe = Probe()
        self.p1EntropyProbe = Probe()
        self.mz2EntropyProbe = Probe()
        self.mz2p1KLDivProbe = Probe()
        self.p1EigProbe = Probe()
        self.p1EigERankProbe = Probe()
        self.z1EigProbe = Probe()
        self.z1EigERankProbe = Probe()
        self.mz2EigProbe = Probe()
        self.mz2EigERankProbe = Probe()
        self.p1z1EigAlignProbe = Probe()
        self.p1mz2EigAlignProbe = Probe()

    # Assumes model is in eval mode, all inputs/outputs are detached, and all model/inputs/outputs are on same device
    def update_probes(self, epoch, loss, loss1, loss2, outList):

        loss = loss.detach().cpu().numpy().astype(float)
        if loss1 is not None:
            loss1 = loss1.detach().cpu().numpy().astype(float)
        if loss2 is not None:
            loss2 = loss2.detach().cpu().numpy().astype(float)

        # Define inputs for metric probes
        # Note that p1, z1, and mz2 are L2 normd, as SimSiam, BYOL, InfoNCE, and MEC use L2 normalized encodings
        # This is taken care of in loss functions, but I have to do it explicitly here
        # This probe update is inaccurate for softmax-normalized encs (DINO, SwAV) or batch normalized encs (Barlow Twins)
        p1 = outList[0][0].detach().cpu().numpy().astype(float)
        p1 /= np.linalg.norm(p1, axis=-1, keepdims=True)
        z1 = outList[0][1].detach().cpu().numpy().astype(float)
        z1 /= np.linalg.norm(z1, axis=-1, keepdims=True)
        r1 = outList[0][2].detach().cpu().numpy().astype(float)
        r2 = outList[1][2].detach().cpu().numpy().astype(float)
        mz2 = outList[1][3].detach().cpu().numpy().astype(float)
        mz2 /= np.linalg.norm(mz2, axis=-1, keepdims=True)

        self.epochProbe.store(epoch)

        # Probe loss throughout training
        self.lossProbe.store(loss)
        self.loss1Probe.store(loss1)
        self.loss2Probe.store(loss2)

        # Get cosine similarity between encodings
        self.r1r2AugSimProbe.store(np.mean(AU.cos_sim_bt_vecs(r1, r2)))
        self.r1AugSimProbe.store(np.mean(AU.cos_sim_within_array(r1)))
        self.p1mz2AugSimProbe.store(np.mean(AU.cos_sim_bt_vecs(p1, mz2)))
        self.p1AugSimProbe.store(np.mean(AU.cos_sim_within_array(p1)))
        # Get concentration (like inverse uniformity) between encodings
        # Equation derived from Uniformity measure in section 4.1.2 (page 5) of Wang/Isola for t=2: https://arxiv.org/abs/2005.10242
        self.r1r2AugConcProbe.store(np.mean(np.exp(4 * AU.cos_sim_bt_vecs(r1, r2) - 4)))
        self.r1AugConcProbe.store(np.mean(np.exp(4 * AU.cos_sim_within_array(r1) - 4)))
        # Representation variance (complete collapse measure)
        self.r1VarProbe.store(np.mean(np.var(r1, axis=0)))
        # Representation correlation strength (off-diag corr values)
        r1Corr = AU.cross_corr(r1, r1)
        self.r1CorrStrProbe.store(np.mean(np.abs(r1Corr[np.triu_indices(m=r1Corr.shape[0], k=1, n=r1Corr.shape[1])])))
        # Mutual information between views using InfoNCE bound
        self.r1r2InfoBoundProbe.store(AU.infonce_bound(r1, r2))
        # Representation encoding correlation ERank
        r1Eigvals, _ = AU.array_eigdecomp(r1, covOrCor='cor')
        # self.r1EigProbe.store(r1Eigvals)
        self.r1EigProbe.storeList = [r1Eigvals]  # Overwrite rather than append (for memory)
        self.r1EigERankProbe.store(np.exp(AU.entropy(r1Eigvals / np.sum(r1Eigvals))))

        # Get entropy and KL div between encodings
        #self.p1EntropyProbe.store(np.mean(AU.cross_ent_bt_vecs(p1, p1)))
        #self.mz2EntropyProbe.store(np.mean(AU.cross_ent_bt_vecs(mz2, mz2)))
        #self.mz2p1KLDivProbe.store(np.mean(AU.kl_div_bt_vecs(mz2, p1)))
        # NOTE: I'm overwriting these with None for now, normalized inputs always give the same values, so no point
        self.p1EntropyProbe.store(None)
        self.mz2EntropyProbe.store(None)
        self.mz2p1KLDivProbe.store(None)

        # Probe encoding correlation stats
        p1Eigvals, _ = AU.array_eigdecomp(p1, covOrCor='cor')
        #self.p1EigProbe.store(p1Eigvals)
        self.p1EigProbe.storeList = [p1Eigvals] # Overwrite rather than append (for memory)
        self.p1EigERankProbe.store(np.exp(AU.entropy(p1Eigvals / np.sum(p1Eigvals))))
        z1Eigvals, z1Eigvecs = AU.array_eigdecomp(z1, covOrCor='cor')
        #self.z1EigProbe.store(z1Eigvals)
        self.z1EigProbe.storeList = [z1Eigvals]  # Overwrite rather than append (for memory)
        self.z1EigERankProbe.store(np.exp(AU.entropy(z1Eigvals / np.sum(z1Eigvals))))
        mz2Eigvals, mz2Eigvecs = AU.array_eigdecomp(mz2, covOrCor='cor')
        #self.mz2EigProbe.store(mz2Eigvals)
        self.mz2EigProbe.storeList = [mz2Eigvals]  # Overwrite rather than append (for memory)
        self.mz2EigERankProbe.store(np.exp(AU.entropy(mz2Eigvals / np.sum(mz2Eigvals))))

        # Probe encoding correlation alignment
        # This method of alignment was used in Zhuo 2023 paper on Rank Differential Mechanism
        p1Corr = AU.cross_corr(p1, p1)
        z1EigvecTrans = np.tensordot(p1Corr, z1Eigvecs, axes=(-1, 0))
        mz2EigvecTrans = np.tensordot(p1Corr, mz2Eigvecs, axes=(-1, 0))
        self.p1z1EigAlignProbe.store(np.real(np.mean(AU.cos_sim_bt_vecs(z1EigvecTrans[:, :512].T, z1Eigvecs[:, :512].T))))
        self.p1mz2EigAlignProbe.store(np.real(np.mean(AU.cos_sim_bt_vecs(mz2EigvecTrans[:, :512].T, mz2Eigvecs[:, :512].T))))

    def write_probes(self, epoch, filePrefix):

        writer = csv.writer(open('Pretrain_Output_{}_{:04d}.csv'.format(filePrefix, epoch), 'w', newline=''))
        writer.writerow(['Loss'] + self.lossProbe.storeList)
        writer.writerow(['Loss1'] + self.loss1Probe.storeList)
        writer.writerow(['Loss2'] + self.loss2Probe.storeList)
        writer.writerow(['Pos Rep Sim'] + self.r1r2AugSimProbe.storeList)
        writer.writerow(['Neg Rep Sim'] + self.r1AugSimProbe.storeList)
        writer.writerow(['Pos Prj Sim'] + self.p1mz2AugSimProbe.storeList)
        writer.writerow(['Neg Prj Sim'] + self.p1AugSimProbe.storeList)
        writer.writerow(['Pos Rep Conc'] + self.r1r2AugConcProbe.storeList)
        writer.writerow(['Neg Rep Conc'] + self.r1AugConcProbe.storeList)
        writer.writerow(['Rep Var'] + self.r1VarProbe.storeList)
        writer.writerow(['Rep Corr Str'] + self.r1CorrStrProbe.storeList)
        writer.writerow(['Rep Mut Info'] + self.r1r2InfoBoundProbe.storeList)
        writer.writerow(['Rep E-Rank'] + self.r1EigERankProbe.storeList)
        writer.writerow(['Prd Entropy'] + self.p1EntropyProbe.storeList)
        writer.writerow(['MomPrj Entropy'] + self.mz2EntropyProbe.storeList)
        writer.writerow(['Prd-MomPrj KL Div'] + self.mz2p1KLDivProbe.storeList)
        writer.writerow(['Prd E-Rank'] + self.p1EigERankProbe.storeList)
        writer.writerow(['Prj E-Rank'] + self.z1EigERankProbe.storeList)
        writer.writerow(['MomPrj E-Rank'] + self.mz2EigERankProbe.storeList)
        writer.writerow(['Prd-Prj Eig Align'] + self.p1z1EigAlignProbe.storeList)
        writer.writerow(['Prd-MomPrj Eig Align'] + self.p1mz2EigAlignProbe.storeList)
        writer.writerow(['Prd Eig'] + np.log(self.p1EigProbe.storeList[-1]).tolist())
        writer.writerow(['MomPrj Eig'] + np.log(self.mz2EigProbe.storeList[-1]).tolist())
        writer.writerow(['Rep Eig'] + np.log(self.r1EigProbe.storeList[-1]).tolist())

    # def plot_probes(self):
    #
    #     xVals = range(len(self.lossProbe.storeList))
    #
    #     plt.plot(xVals, self.lossProbe.storeList)
    #     plt.xlabel('Epoch')
    #     plt.ylabel('Loss')
    #     plt.grid(visible=True, which='major', axis='x')
    #     plt.show()
    #
    #     plt.plot(xVals, self.r1r2AugSimProbe.storeList, label='r1r2-SameSrc')
    #     plt.plot(xVals, self.r1AugSimProbe.storeList, label='r1-DiffSrc')
    #     plt.xlabel('Epoch')
    #     plt.ylabel('Average Cosine Similarity')
    #     plt.grid(visible=True, which='major', axis='x')
    #     plt.show()


class Evaluation_Probes:

    def __init__(self):

        self.ptEpProbe = Probe()
        self.repEigProbe = Probe()
        self.repEigERankProbe = Probe()
        self.repLolipProbe = Probe()
        self.normRepLolipProbe = Probe()
        self.cosKnnAccProbe = Probe()
        self.mseKnnAccProbe = Probe()
        self.clnTrainAccProbe = Probe()
        self.clnTestAccProbe = Probe()
        self.advAccProbe = Probe()
        #self.atkVecEntProbe = Probe()

    def update_probes(self, ptEp, repBank, avgRepLolip, avgNormRepLolip, cosKnnAcc, mseKnnAcc, clnTrainAcc, clnTestAcc, advAcc):
        repBank = repBank.cpu().numpy()

        self.ptEpProbe.store(ptEp)

        repEigvals, _ = AU.array_eigdecomp(repBank, covOrCor='cor')
        #self.repEigProbe.storeList = [repEigvals]
        self.repEigProbe.store(repEigvals)
        self.repEigERankProbe.store(np.exp(AU.entropy(repEigvals / np.sum(repEigvals))))
        self.repLolipProbe.store(avgRepLolip)
        self.normRepLolipProbe.store(avgNormRepLolip)

        self.cosKnnAccProbe.store(cosKnnAcc)
        self.mseKnnAccProbe.store(mseKnnAcc)
        self.clnTrainAccProbe.store(clnTrainAcc)
        self.clnTestAccProbe.store(clnTestAcc)
        self.advAccProbe.store(advAcc)

        # Measure cossim of every atk vector to a random vector, count vectors with the same cossim, calculate entropy
        # Note perturbTens is np.float32, so it's necessary to convert the rand vector to np.float32
        # If rand vector is np.float64, tiny differences (1e-9) appear and screw up np.unique()
        #perturbTens = perturbTens.cpu().numpy()
        #simVals = AU.cos_sim_bt_arrays(np.random.rand(1, perturbTens.shape[1]).astype(np.float32), perturbTens)
        #_, counts = np.unique(simVals, return_counts=True)
        #entropy = 0
        #for count in counts:
        #    entropy -= count / len(simVals) * np.log(count / len(simVals))
        #self.atkVecEntProbe.store(entropy)

    def write_probes(self):

        writer = csv.writer(open('Evaluate_Output.csv', 'w', newline=''))
        for eigArr in self.repEigProbe.storeList:
            writer.writerow(['Rep Eig'] + np.log(eigArr).tolist())
        writer.writerow(['Rep E-Rank'] + self.repEigERankProbe.storeList)
        writer.writerow(['Rep Local Lip'] + self.repLolipProbe.storeList)
        writer.writerow(['Normalized Rep Local Lip'] + self.normRepLolipProbe.storeList)
        writer.writerow(['Cosine KNN Acc'] + self.cosKnnAccProbe.storeList)
        writer.writerow(['MSE KNN Acc'] + self.mseKnnAccProbe.storeList)
        writer.writerow(['Train Acc'] + self.clnTrainAccProbe.storeList)
        writer.writerow(['Test Acc'] + self.clnTestAccProbe.storeList)
        writer.writerow(['Adv Acc'] + self.advAccProbe.storeList)

    # def plot_probes(self):
    #
    #     xVals = self.ptEpProbe.storeList
    #
    #     plt.plot(xVals, self.clnTrainAccProbe.storeList)
    #     plt.xlabel('Pretrain Epoch')
    #     plt.ylabel('Clean Cls Acc')
    #     plt.grid(visible=True, which='major', axis='x')
    #     plt.show()
    #
    #     plt.plot(xVals, self.clnTestAccProbe.storeList)
    #     plt.xlabel('Pretrain Epoch')
    #     plt.ylabel('Clean Cls Acc')
    #     plt.grid(visible=True, which='major', axis='x')
    #     plt.show()
