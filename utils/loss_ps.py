import torch
import torch.nn as nn
import numpy as np


from utils.metrics import bbox_iou, box_iou, bbox_alpha_iou
from utils.torch_utils import de_parallel, is_parallel
from utils.general import xywh2xyxy
import torch.nn.functional as F

class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss

def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class ComputeLoss_v4:
    # Compute losses
    def __init__(self, model, autobalance=False):
        self.sort_obj_iou = False
        h = model.hyp  # hyperparameters
        self.model = model
        # def compute_loss_v4(p, targets, model):  # predictions, targets, model
    def __call__(self, p, targets):  # predictions, targets, model
        device = targets.device
        ft = torch.cuda.FloatTensor if p[0].is_cuda else torch.Tensor
        lcls, lbox, lobj = ft([0]).to(device), ft([0]).to(device), ft([0]).to(device)
        tcls, tbox, indices, anchors = self.build_targets(p, targets, self.model)  # targets
        red = 'mean'  # Loss reduction (sum or mean)

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=ft([1.0]), reduction=red).to(device)
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=ft([1.0]), reduction=red).to(device)

        # class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        cp, cn = smooth_BCE(eps=0.0)

        # focal loss
        g = 0.0  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        # per output
        nt = 0  # number of targets
        np = len(p)  # number of outputs
        balance = [4.0, 1.0, 0.4] if np == 3 else [4.0, 1.0, 0.4, 0.1]  # P3-5 or P3-6
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0]).to(device)  # target obj

            nb = b.shape[0]  # number of targets
            if nb:
                nt += nb  # cumulative targets
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

                # GIoU
                pxy = ps[:, :2].sigmoid() * 2. - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1).to(device)  # predicted box
                giou = bbox_iou(pbox.t(), tbox[i], x1y1x2y2=False, GIoU=True)  # giou(prediction, target)
                lbox += (1.0 - giou).sum() if red == 'sum' else (1.0 - giou).mean()  # giou loss

                # Obj
                tobj[b, a, gj, gi] = (1.0 - self.model.gr) + self.model.gr * giou.detach().clamp(0).type(tobj.dtype)  # giou ratio

                # Class
                if self.model.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(ps[:, 5:], cn).to(device)  # targets
                    t[range(nb), tcls[i]] = cp
                    lcls += BCEcls(ps[:, 5:], t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            lobj += BCEobj(pi[..., 4], tobj) * balance[i]  # obj loss

        s = 3 / np  # output count scaling
        lbox *= 0.05 * s
        lobj *= 1.0 * s * (1.4 if np == 4 else 1.)
        lcls *= 0.5 * s
        bs = tobj.shape[0]  # batch size
        if red == 'sum':
            g = 3.0  # loss gain
            lobj *= g / bs
            if nt:
                lcls *= g / nt / self.model.nc
                lbox *= g / nt

        loss = lbox + lobj + lcls
        return loss * bs, torch.cat((lbox, lobj, lcls)).detach()


    def build_targets(self, p, targets, model):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        det = model.module.model[-1] if type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel) \
            else model.model[-1]  # Detect() module
        na, nt = det.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(6, device=targets.device)  # normalized to gridspace gain
        off = torch.tensor([[1, 0], [0, 1], [-1, 0], [0, -1]], device=targets.device).float()  # overlap offsets
        at = torch.arange(na).view(na, 1).repeat(1, nt)  # anchor tensor, same as .repeat_interleave(nt)

        g = 0.5  # offset
        style = 'rect4'
        for i in range(det.nl):
            anchors = det.anchors[i]
            gain[2:] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            a, t, offsets = [], targets * gain, 0
            if nt:
                r = t[None, :, 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1. / r).max(2)[0] < model.hyp['anchor_t']  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n) = wh_iou(anchors(3,2), gwh(n,2))
                a, t = at[j], t.repeat(na, 1, 1)[j]  # filter

                # overlaps
                gxy = t[:, 2:4]  # grid xy
                z = torch.zeros_like(gxy)
                if style == 'rect2':
                    j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                    a, t = torch.cat((a, a[j], a[k]), 0), torch.cat((t, t[j], t[k]), 0)
                    offsets = torch.cat((z, z[j] + off[0], z[k] + off[1]), 0) * g
                elif style == 'rect4':
                    j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                    l, m = ((gxy % 1. > (1 - g)) & (gxy < (gain[[2, 3]] - 1.))).T
                    a, t = torch.cat((a, a[j], a[k], a[l], a[m]), 0), torch.cat((t, t[j], t[k], t[l], t[m]), 0)
                    offsets = torch.cat((z, z[j] + off[0], z[k] + off[1], z[l] + off[2], z[m] + off[3]), 0) * g

            # Define
            b, c = t[:, :2].long().T  # image, class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices

            # Append
            indices.append((b, a, gj, gi))  # image, anchor, grid indices
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch