from numpy.ma.core import outer
from pandas._libs.tslibs.conversion import OutOfBoundsTimedelta
import torch
import torch.nn as nn
from transformers.utils.dummy_pt_objects import MODEL_FOR_VISION_2_SEQ_MAPPING
from model_new import Transformer, Transformer2, Transformer3
from config import *
import torch.nn.functional as F



class cca_loss():
    def __init__(self, outdim_size, use_all_singular_values, device):
        self.outdim_size = outdim_size
        self.use_all_singular_values = use_all_singular_values
        self.device = device
        # print(device)

    def loss(self, H1, H2):
        """
        It is the loss function of CCA as introduced in the original paper. There can be other formulations.
        """

        r1 = 1e-3
        r2 = 1e-3
        eps = 1e-9

        H1, H2 = H1.t(), H2.t()
        # assert torch.isnan(H1).sum().item() == 0
        # assert torch.isnan(H2).sum().item() == 0

        o1 =  H1.size(0)
        o2 = H2.size(0)

        m = H1.size(1)
#         print(H1.size())

        H1bar = H1 - H1.mean(dim=1).unsqueeze(dim=1)
        H2bar = H2 - H2.mean(dim=1).unsqueeze(dim=1)
        # assert torch.isnan(H1bar).sum().item() == 0
        # assert torch.isnan(H2bar).sum().item() == 0

        SigmaHat12 = (1.0 / (m - 1)) * torch.matmul(H1bar, H2bar.t())
        SigmaHat11 = (1.0 / (m - 1)) * torch.matmul(H1bar,
                                                    H1bar.t()) + r1 * torch.eye(o1, device=self.device)
        SigmaHat22 = (1.0 / (m - 1)) * torch.matmul(H2bar,
                                                    H2bar.t()) + r2 * torch.eye(o2, device=self.device)
        # assert torch.isnan(SigmaHat11).sum().item() == 0
        # assert torch.isnan(SigmaHat12).sum().item() == 0
        # assert torch.isnan(SigmaHat22).sum().item() == 0

        # Calculating the root inverse of covariance matrices by using eigen decomposition
        [D1, V1] = torch.symeig(SigmaHat11, eigenvectors=True)
        [D2, V2] = torch.symeig(SigmaHat22, eigenvectors=True)
        # assert torch.isnan(D1).sum().item() == 0
        # assert torch.isnan(D2).sum().item() == 0
        # assert torch.isnan(V1).sum().item() == 0
        # assert torch.isnan(V2).sum().item() == 0

        # Added to increase stability
        posInd1 = torch.gt(D1, eps).nonzero()[:, 0]
        D1 = D1[posInd1]
        V1 = V1[:, posInd1]
        posInd2 = torch.gt(D2, eps).nonzero()[:, 0]
        D2 = D2[posInd2]
        V2 = V2[:, posInd2]
        # print(posInd1.size())
        # print(posInd2.size())

        SigmaHat11RootInv = torch.matmul(
            torch.matmul(V1, torch.diag(D1 ** -0.5)), V1.t())
        SigmaHat22RootInv = torch.matmul(
            torch.matmul(V2, torch.diag(D2 ** -0.5)), V2.t())

        Tval = torch.matmul(torch.matmul(SigmaHat11RootInv,
                                         SigmaHat12), SigmaHat22RootInv)
#         print(Tval.size())

        if self.use_all_singular_values:
            # all singular values are used to calculate the correlation
            tmp = torch.matmul(Tval.t(), Tval)
            corr = torch.trace(torch.sqrt(tmp))
            # assert torch.isnan(corr).item() == 0
        else:
            # just the top self.outdim_size singular values are used
            trace_TT = torch.matmul(Tval.t(), Tval)
            trace_TT = torch.add(trace_TT, (torch.eye(trace_TT.shape[0])*r1).to(self.device)) # regularization for more stability
            U, V = torch.symeig(trace_TT, eigenvectors=True)
            U = torch.where(U>eps, U, (torch.ones(U.shape).float()*eps).to(self.device))
            U = U.topk(self.outdim_size)[0]
            corr = torch.sum(torch.sqrt(U))
        return -corr


class DeepCCA(nn.Module):
    def __init__(self, model1, model2, outdim_size, use_all_singular_values, device=torch.device('cuda')):
        super(DeepCCA, self).__init__()
        self.model1 = model1
        self.model2 = model2
        # self.classifier = nn.Linear(6, 3)
       
        self.loss = cca_loss(outdim_size, use_all_singular_values, device).loss

    def forward(self, x1, x2):
        
        output1 = self.model1(x1)
        output2 = self.model2(x2)
        
        

        return output1, output2


class DeepCCA_fusion(nn.Module):
    def __init__(self, model1, outdim_size, use_all_singular_values, d_feature, d_model, d_inner,
            n_layers, n_head, d_k=64, d_v=64, dropout = 0.5,
            class_num=3, device=torch.device('cuda')):
        super(DeepCCA_fusion, self).__init__()
        self.model1 = model1
        self.Transformer = Transformer3(device=device, d_feature=6, d_model=d_model, d_inner=d_inner,
                            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v, dropout=dropout, class_num=class_num)

        self.loss = cca_loss(outdim_size, use_all_singular_values, device).loss
        self.classifier = nn.Linear(6, class_num)

    def forward(self, x1, x2):
        
        # feature * batch_size
        x1, x2 = self.model1(x1, x2)
        x = torch.cat((x1, x2), dim = 1)
        # out = self.classifier(F.relu(x))
        out = self.Transformer(x)

        return out, x1, x2


