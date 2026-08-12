import json
import matplotlib.pyplot as plt

# Il JSON fornito dal tuo output
gt_json = """
    paste here the JSON information of the wanted frame
"""

def plot_bev_map(data):
    fig, ax = plt.subplots(figsize=(10, 12))
    
    # Mappatura dei colori per Matplotlib
    color_map = {
        "yellow": "gold",         
        "blue": "blue",
        "orange_small": "darkorange",
        "orange_big": "darkorange", 
        "red": "red"                
    }

    # Disegniamo l'Ego Vehicle all'origine (0,0)
    ax.plot(0, 0, marker='s', color='black', markersize=12, label='Ego Vehicle')
    
    # Piccola freccia per indicare la direzione di marcia dell'auto (Avanti = asse X)
    ax.arrow(0, 0, 0, 2, head_width=1, head_length=1.5, fc='black', ec='black')

    for cone in data['cones']:
        # In CARLA Right-Handed (come configurato in lidar_points_to_rh):
        # X è in avanti, Y è a sinistra.
        x, y, z = cone['position']
        cone_class = cone['class']
        num_pts = cone['num_lidar_points']
        c_id = cone['instance_id']

        color = color_map.get(cone_class, "gray")
        
        # Rendiamo semi-trasparente il cono se il LiDAR non lo ha "visto"
        alpha = 1.0 if num_pts > 0 else 0.25

        # Disegniamo il cono come un triangolo ('^')
        # Plottiamo Y sull'asse orizzontale e X su quello verticale
        ax.scatter(y, x, c=color, marker='^', s=150, alpha=alpha, edgecolors='black', linewidth=1)

        # Aggiungiamo le info sotto il marker
        ax.text(y, x - 0.7, f"ID: {c_id}\npts: {num_pts}", 
                fontsize=8, ha='center', va='top', color='black', 
                bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1))

    # --- Configurazione Grafico ---
    
    ax.set_title(f"LiDAR Ground Truth - Bird's Eye View (Frame {data.get('frame', 0)})", fontsize=14, fontweight='bold')
    ax.set_xlabel("Y (Sinistra/Destra) [metri]", fontsize=12)
    ax.set_ylabel("X (Avanti) [metri]", fontsize=12)
    
    # Invertiamo l'asse X del grafico (che per noi è l'asse Y delle coordinate) 
    # affinché i valori positivi di Y (sinistra) appaiano effettivamente a sinistra sul grafico.
    ax.invert_xaxis()
    
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.axis('equal') # Mantiene le proporzioni geometriche reali
    
    # Aggiungiamo un margine alla vista per non tagliare fuori nulla
    ax.margins(0.1)

    # Legenda personalizzata
    handles = [
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='black', markersize=10, label='Ego Vehicle'),
        plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='gold', markeredgecolor='black', markersize=10, label='Yellow Cone'),
        plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='blue', markeredgecolor='black', markersize=10, label='Blue Cone'),
        plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='darkorange', markeredgecolor='black', markersize=10, label='Orange Cone'),
        plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='gray', markeredgecolor='black', markersize=10, alpha=0.3, label='0 LiDAR Points')
    ]
    ax.legend(handles=handles, loc='upper left')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Carichiamo la stringa come dizionario Python
    data = json.loads(gt_json)
    plot_bev_map(data)