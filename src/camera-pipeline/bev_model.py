from pathlib import Path
from typing import Dict, Optional
 
import timm
import torch
import torch.nn as nn
 
from model import DetectionHead2d 


class CameraBEVNet(nn.Model):

    #ramo per camera: backbone HRNet, lifting geometrico, bev pooling, head bev (temporanea)
    #il gradiente della loss bev risale fino al backbone

    def __init__(
        self,
        cgf,
        pretrained: bool = True,
        backbone_checkpoint_path: Optional[Path] = None
    ):
        super().__init__()
        self.cgf = cgf

        #carico backbone da timm con features only
        self.backbone = timm.create_model(
            cgf.backbone_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(cgf.feature_index,),
        )


        feature_channels = self.backbone.feature_info.channels()[0]
        print(f"[BEV] Backbone {cfg.backbone_name}: {feature_channels} canali, " f"stride {self.backbone.feature_info.reduction()[0]}")

        if backbone_checkpoint_path is not None:
            self._load_warmup_backbone(Path(backbone_checkpoint_path))

        
        self.head = DetectionHead2d(
            in_channels=feature_channels,
            hidden_channels=cfg.head_hidden_channels,
            num_classes=cfg.num_classes,
            numlayers=cfg.head_num_layers,
        )

    def _load_warmup_backbone(self, checkpoint_path: Path):

        #caricamento pesi del backbone da checkpoint di warmup
        #salvato da warmup_training.save_backbone()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        
        state = torch.load(checkpoint_path, map_location="cpu")["backbone_state_dict"]
        stripped = {k.replace("backbone.", "", 1): v for k, v in state.items()}
        missing, unexpected = self.backbone.load_state_dict(stripped, strict=False)

        if missing or unexpected:
            print(f"[BEV] Warning: Missing keys in backbone state_dict: {missing}")
            print(f"[BEV] Warning: Unexpected keys in backbone state_dict: {unexpected}")

    
    def lift_to_bev(self, features_map, depth, k, t):

        # feature_map (B, C, Hf, Wf) B=batch size, C=channels, Hf=height feature map, Wf=width feature map
        # depth (B, H, W) B=batch size, H=height image, W=width image
        # K (B, 4) B=batch size, 4=[fx, fy, cx, cy] fx=focal length x, fy=focal length y, cx=principal point x, cy=principal point y
        # T (B, 4, 4) B=batch size, 4x4 matrice di trasformazione da camera a coordinate mondo
        B, C, Hf, Wf = features_map.shape
        cfg = self.cgf
        Hb, Wb = cfg.bev_H, cfg.bev_W
        s = cfg.feature_stride
        device = features_map.device


        #depth al centro del pixel di ogni cella della feature map

        #creazione della griglia di coordinate pixel della feature map
        us = torch.arange(Wf, device=device, dtype=torch.float32) * s + s / 2.0
        vs = torch.arange(Hf, device=device, dtype=torch.float32) * s + s / 2.0
        
        #prende i valori di depth corrispondenti alle coordinate della feature map, clamp per evitare out of bounds
        d = depth.index_select(1, vs.long().clamp(max=depth.shape[1] - 1)) \
                 .index_select(2, us.long().clamp(max=depth.shape[2] - 1))
        #il risultato e' la matrice d dove ogni pixel ha la distanza stimata

        #proiezione su piano tridimensionale delle feature della matrice d
        #usiamo la formula di proiezione inversa: X = (u - cx) * Z / fx, Y = (v - cy) * Z / fy, Z = d
        #usiamo le matrici intrinseche K e la trasformazione T per ottenere le coordinate mondo
        fx, fy, cx, cy = (K[:, i].view(B, 1, 1) for i in range(4))
        x_cam = (us.view(1, 1, Wf) - cx) * d / fx
        y_cam = (vs.view(1, Hf, 1) - cy) * d / fy
        pts = torch.stack([x_cam, y_cam.expand_as(d), d], dim=1).reshape(B, 3, -1)


        #spostamento del sistema di riferimento dalla camera al centro della macchina usando matrici di trasformazione
        
        #METTERSI D'ACCORDO CON ANDRE SUL SISTEMA DI RIFERIMENTO!!!!!!!!!!!!!!!!!!!!!!
        pts_v = torch.bmm(T[:, :3, :3], pts) + T[:, :3, 3:4]
        x_v, y_v = pts_v[:, 0], pts_v[:, 1]
        #METTERSI D'ACCORDO CON ANDRE SUL SISTEMA DI RIFERIMENTO!!!!!!!!!!!!!!!!!!!!!!


        #calcolo delle celle bev
        #prima conversiamo le distanxe in metri (x, y) negli indici riga e colonna della griglia bev
        rows = torch.floor((cfg.x_max - x_v) / cfg.resolution).long()
        cols = torch.floor((cfg.y_max - y_v) / cfg.resolution).long()

        d_flat = d.reshape(B, -1)
        #eliminazione dei punti fuori dalla griglia bev o con profondità non valida con maschera
        valid = ((d_flat > 0) & torch.isfinite(d_flat)
                & (rows >= 0) & (rows < Hb)
                & (cols >= 0) & (cols < Wb)).reshape(-1)
        

        #FLATTEENING
        N = rows.shape[1]
        batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, N)
        lin = ((batch_idx * Hb + rows.clamp(0, Hb - 1)) * Wb + cols.clamp(0, Wb - 1)).reshape(-1)

        feats = feature_map.reshape(B, C, -1).permute(0, 2, 1).reshape(-1, C)
        idx = lin[valid]


        #costruzione fisica della mappa bev
        bev = feature_map.new_zeros(B * Hb * Wb, C)
        counts = feature_map.new_zeros(B * Hb * Wb, 1)

        #prende le info visive dei punti validi e le mette nelle corrispondenti celle
        #se piu punti finiscono nella stessa cella le informazioni si sommano
        bev.index_add_(0, idx, feats[valid])
        counts.index_add_(0, idx, torch.ones(idx.shape[0], 1, device=device, dtype=bev.dtype))
        #fa la media evitando che le celle con piu punti abbiano valori piu alti
        bev = bev / counts.clamp(min=1.0)

        return bev.reshape(B, Hb, Wb, C).permute(0, 3, 1, 2).contiguous()
        

    def forward(self, images, depth, K, T) -> Dict[str, torch.Tensor]:
        feature_map = self.backbone(images)[0]
        bev_features = self.lift_to_bev(feature_map, depth, K, T)
        heatmap_logits, offset_pred = self.head(bev_features)
        return {
            "heatmap_logits": heatmap_logits,
            "offset_pred": offset_pred,
            "bev_features": bev_features,
        }


    def get_param(self, backbone_lr: float, head_lr: float, weight_decay: float):
        #stessa interfaccia del warmup
        return [
            {"params": self.backbone.parameters(), "lr": backbone_lr, "weight_decay": weight_decay},
            {"params": self.head.parameters(), "lr": head_lr, "weight_decay": weight_decay},
        ]

