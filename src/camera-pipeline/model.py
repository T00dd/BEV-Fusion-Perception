from typing import Dict, Tuple

import timm
import torch
import torch.nn as nn


class DetectionHead2d(nn.Module):
    #head leggera per detection 2d
    #prende in input la feature map creata dal backbone e produce:
    # -heatmap con picchi nei centri degli oggetti
    # -offset a due canali per la localizzazione

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_classes: int = 2,
        numlayers: int = 3,
    ):
        super().__init__()

        #backbone leggera
        layers = []
        current_channels = in_channels
        for _ in range(numlayers-1):
            layers.append(nn.Conv2d(current_channels, hidden_channels, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(hidden_channels))
            layers.append(nn.ReLU(inplace=True))
            current_channels = hidden_channels
        self.trunk = nn.Sequential(*layers)

        #head per heatmap
        self.heatmap_head = nn.Conv2d(current_channels, num_classes, kernel_size=1)

        #head per offset
        self.offset_head = nn.Conv2d(current_channels, 2, kernel_size=1)


        prior_prob = 0.01
        bias_value = -torch.log(torch.tensor((1 - prior_prob) / prior_prob))
        nn.init.constant_(self.heatmap_head.bias, bias_value.item())
        nn.init.normal_(self.heatmap_head.weight, std=0.01)

        nn.init.normal_(self.offset_head.weight, std=0.01)
        nn.init.zeros_(self.heatmap_head.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x= self.trunk(x)
        heatmap = self.heatmap_head(x)
        offset = self.offset_head(x)
        return heatmap, offset
    

class HRNet_with_detection_head(nn.Module):
    
    #modello completo con backbone HRNet e head per detection 2d per il warmup

    def __init__(
        self, 
        backbone_name: str = 'hrnet_w32.ms_in1k',
        feature_index: int = 1,
        num_classes: int = 2,
        head_hidden_channels: int = 64,
        head_numlayers: int = 3,
        pretrained: bool = True,
    ):
        super().__init__()
        self.feature_index = feature_index
    
        #carico backbone da timm con features only
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(feature_index,),
        )
    
        feature_channels = self.backbone.feature_info.channels()[0]
        feature_reduction = self.backbone.feature_info.reduction()[0]
    
        print(f"Backbone {backbone_name} loaded with feature channels: {feature_channels} and stride: {feature_reduction}")
        print(f"Using feature index: {feature_index}")

        #head per detection 2d
        self.detection_head = DetectionHead2d(
            in_channels=feature_channels,
            hidden_channels=head_hidden_channels,
            num_classes=num_classes,
            numlayers=head_numlayers,
        )

    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:

        #backbone: ritorna una lista con un solo elemento (la feature map scelta)
        features = self.backbone(x)
        feature_map = features[0]  #(B, C_feat, H/stride, W/stride)

        heatmap_logits, offset_pred = self.head(feature_map)

        return{

            "heatmap_logits": heatmap_logits,
            "offset_pred": offset_pred,
            "feature_map": feature_map, #returniamo anche le feature per futura visualizzazione oopure debug
        }
    

    def get_param(self, backbone_lr: float, head_lr: float, weight_decay: float):

        #ritorna i parametri del modello con learning rate differenziati per backbone e head

        return [
            {"params": self.backbone.parameters(), "lr": backbone_lr, "weight_decay": weight_decay},
            {"params": self.detection_head.parameters(), "lr": head_lr, "weight_decay": weight_decay},
        ]

        