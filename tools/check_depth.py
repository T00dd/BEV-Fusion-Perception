import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

def verify_depth_map(depth_npy_path, rgb_image_path=None, max_vis_depth=100.0):
    """
    Carica e visualizza la mappa di profondità .npy generata da CARLA, 
    confrontandola (se disponibile) con l'immagine RGB corrispondente.
    """
    if not os.path.exists(depth_npy_path):
        print(f"Errore: File depth non trovato -> {depth_npy_path}")
        return

    # 1. Caricamento dei dati di profondità (in metri)
    depth_m = np.load(depth_npy_path)
    
    print("--- Statistiche Mappa di Profondità ---")
    print(f"Risoluzione : {depth_m.shape[1]}x{depth_m.shape[0]}")
    print(f"Distanza Min: {np.min(depth_m):.2f} m")
    print(f"Distanza Max: {np.max(depth_m):.2f} m")
    print(f"Distanza Med: {np.mean(depth_m):.2f} m")
    print("---------------------------------------")

    # 2. Setup della visualizzazione
    # Limitiamo la profondità massima visiva (clipping) per non farsi rovinare
    # la scala dei colori dal cielo (che in CARLA è tipicamente 1000m)
    depth_vis = np.clip(depth_m, 0, max_vis_depth)

    if rgb_image_path and os.path.exists(rgb_image_path):
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Carica e converti l'immagine da BGR a RGB per Matplotlib
        rgb_img = cv2.imread(rgb_image_path)
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        
        axes[0].imshow(rgb_img)
        axes[0].set_title("Telecamera RGB (Sinistra)", fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        # Visualizzazione Depth
        im = axes[1].imshow(depth_vis, cmap='plasma')
        axes[1].set_title(f"Mappa di Profondità (Clipped a {max_vis_depth}m)", fontsize=14, fontweight='bold')
        axes[1].axis('off')
        
        fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04, label="Distanza (Metri)")
    else:
        plt.figure(figsize=(10, 8))
        im = plt.imshow(depth_vis, cmap='plasma')
        plt.title(f"Mappa di Profondità (Clipped a {max_vis_depth}m)", fontsize=14, fontweight='bold')
        plt.colorbar(im, fraction=0.046, pad=0.04, label="Distanza (Metri)")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # INSERISCI QUI I PERCORSI AI TUOI FILE DI TEST
    # Sostituisci i percorsi con quelli generati nella tua cartella 'beta_dataset'
    
    depth_file = "../beta_dataset/scenes/scene_0000/depth/frame_000000.npy"
    rgb_file = "../beta_dataset/scenes/scene_0000/images/frame_000000_cam_left.png" 
    
    verify_depth_map(depth_file, rgb_file, max_vis_depth=20.0)