
from typing import Dict, Tuple
 
import torch
import torch.nn as nn
import torch.nn.functional as F


class CenterNetFocalLoss(nn.Module):

    #formule:
    #per pixel  con heatmap_gt == 1: L_pos = -(1-p)^alpha * log(p)
    #per pixel con heatmap_gt < 1: L_neg = -(1-heatmap_gt)^beta * p^alpha * log(1-p)


    def __init__(self, alpha: float = 2.0, beta: float = 4.0, eps: float = 1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    def forward(self, heatmap_logits: torch.Tensor, gt_heatmap: torch.Tensor) -> torch.Tensor:

        pred = torch.sigmoid(heatmap_logits).clamp(self.eps, 1-self.eps)

        pos_mask = (gt_heatmap == 1.0).float()
        neg_mask = (gt_heatmap < 1.0).float()

        #loss positivi
        pos_loss = -((1 - pred) ** self.alpha) * torch.log(pred) * pos_mask

        #loss negativi
        neg_loss = -((1 - gt_heatmap) ** self.beta) * (pred ** self.alpha) * torch.log(1 - pred) * neg_mask

        #aggregazione e normalizzazione
        num_pos = pos_mask.sum().clamp(min=1.0)
        loss = (pos_loss.sum() + neg_loss.sum()) / num_pos

        return loss
    

class OffsetL1Loss(nn.Module):

    def forward(
        self,
        offset_pred: torch.Tensor,
        offset_gt: torch.Tensor,
        offset_mask: torch.Tensor,
    ) -> torch.Tensor:
        
        #offset_pred (B, 2, H, W)
        #offset_gt (B, 2, H, W)
        #offset_mask (B, 1, H, W)

        loss_per_pixel = F.l1_loss(offset_pred, offset_gt, reduction="none") #(B, 2, H, W)
        loss_per_pixel = loss_per_pixel * offset_mask #zero fuori dai centri
        
        num_pos = offset_mask.sum().clamp(min=1.0)
        loss = loss_per_pixel.sum() / (num_pos * 2)
        return loss
    


