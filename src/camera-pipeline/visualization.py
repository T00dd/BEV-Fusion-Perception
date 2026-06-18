from pathlib import Path
from typing import List
 
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

#visualizzazione della predizione durante il training

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])

def denormalize_image(image_tensor: torch.Tensor) -> np.ndarray:
    
    #inverte la normalizzazione di imagenet
    img = image_tensor.cpu().numpy().transpose(1, 2, 0)  #(H, W, 3)
    img = img * IMAGENET_STD + IMAGENET_MEAN
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img


def color_heatmap(heatmap: np.ndarray, num_classes: int = 2) -> np.ndarray:

    #converte una heatmap in un immagine rgb

    H, W = heatmap.shape[1], heatmap.shape[2]
    rgb = np.zeros((H, W, 3), dtype=np.float32)
    if num_classes >= 1:
        rgb[..., 2] = heatmap[0]  # blu
    if num_classes >= 2:
        rgb[..., 0] = heatmap[1]  # rosso (giallo = rosso + verde)
        rgb[..., 1] = heatmap[1]  # verde
    rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)
    return rgb


def visualize_predictions(
        image_tensor: torch.Tensor,
        heatmap_gt: torch.Tensor,
        heatmap_pred_logits: torch.Tensor,
        stride: int,
        output_path: Path,
    ):
    
    #crea una ficura con immagine, heatmap gt e heatmap predetta dal modello

    image = denormalize_image(image_tensor)
    H, W = image.shape[:2]

    #upsample heatmap a risoluzione immagine

    gt_up = F.interpolate(
        heatmap_gt.unsqueeze(0).float(),
        size=(H, W),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0).cpu().numpy()

    pred_probs = torch.sigmoid(heatmap_pred_logits).unsqueeze(0).float()
    pred_up = F.interpolate(
        pred_probs,
        size=(H, W),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0).cpu().numpy()

    #conversione heatmap in rgb
    gt_rgb = color_heatmap(gt_up)
    pred_rgb = color_heatmap(pred_up)



def save_visualization_batch(
    images: torch.Tensor,
    heatmaps_gt: torch.Tensor,
    heatmaps_pred_logits: torch.Tensor,
    sample_ids: List[str],
    output_dir: Path,
    stride: int,
    epoch: int,
    max_to_save: int = 8,
):
    
    #salva visualizzazioni per un intero batch
    
    output_dir = Path(output_dir) / f"epoch_{epoch:03d}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    num_to_save = min(max_to_save, images.shape[0])

    for i in range(num_to_save):

        safe_id = sample_ids[i].replace("/", "_")
        out_path = output_dir / f"{safe_id}.png"
        visualize_predictions(
            images[i],
            heatmaps_gt[i],
            heatmaps_pred_logits[i],
            stride=stride,
            output_path=out_path,
        )
