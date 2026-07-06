import json
import matplotlib.pyplot as plt

# Il JSON fornito dal tuo output
gt_json = """
{
  "frame": 0,
  "cones": [
    {"instance_id": 273, "class": "orange_small", "position": [35.081346886580604, 3.6418919097386606, -1.56153077412867], "box": [35.081346886580604, 3.6418919097386606, -1.40153077412867, 0.23, 0.23, 0.32, 0.0], "num_lidar_points": 3, "distance": 35.29771330510871},
    {"instance_id": 275, "class": "orange_small", "position": [10.054846099279871, 2.657721806711379, -1.5109404948237568], "box": [10.054846099279871, 2.657721806711379, -1.3509404948237569, 0.23, 0.23, 0.32, 0.0], "num_lidar_points": 38, "distance": 10.487538104942812},
    {"instance_id": 240, "class": "yellow", "position": [49.19736586043291, 13.420284206476936, -1.5897661899880404], "box": [49.19736586043291, 13.420284206476936, -1.4297661899880405, 0.23, 0.23, 0.32, 0.0], "num_lidar_points": 1, "distance": 51.014988651826016},
    {"instance_id": 239, "class": "yellow", "position": [45.27284499915868, 12.03781994089286, -1.581872798004298], "box": [45.27284499915868, 12.03781994089286, -1.421872798004298, 0.23, 0.23, 0.32, 0.0], "num_lidar_points": 2, "distance": 46.86748686990705},
    {"instance_id": 238, "class": "yellow", "position": [41.39018983923927, 10.628775963133194, -1.5740649532930604], "box": [41.39018983923927, 10.628775963133194, -1.4140649532930605, 0.23, 0.23, 0.32, 0.0], "num_lidar_points": 1, "distance": 42.75649977599751},
    {"instance_id": 237, "class": "yellow", "position": [37.527841907141124, 9.240818968439726, -1.5662974998180914], "box": [37.527841907141124, 9.240818968439726, -1.4062974998180915, 0.23, 0.23, 0.32, 0.0], "num_lidar_points": 2, "distance": 38.67440143134535},
    {"instance_id": 236, "class": "yellow", "position": [33.667726153656645, 7.9140846383998, -1.558532572313112], "box": [33.667726153656645, 7.9140846383998, -1.3985325723131121, 0.23, 0.23, 0.32, 0.0], "num_lidar_points": 3, "distance": 34.61364490164525},
    {"instance_id": 235, "class": "yellow", "position": [29.795612925378578, 6.681817283997134, -1.550740307675568], "box": [29.795612925378578, 6.681817283997134, -1.390740307675568, 0.23, 0.23, 0.32, 0.0], "num_lidar_points": 4, "distance": 30.56729282123465},
    {"instance_id": 234, "class": "yellow", "position": [25.90177672781838, 5.571081064340802, -1.5428848955488377], "box": [25.90177672781838, 5.571081064340802, -1.3828848955488378, 0.23, 0.23, 0.32, 0.0], "num_lidar_points": 4, "distance": 26.530196993568435},
    {"instance_id": 233, "class": "yellow", "position": [21.980649573315986, 4.602084653482052, -1.5350001931778934], "box": [21.980649573315986, 4.602084653482052, -1.3750001931778935, 0.23, 0.23, 0.32, 0.0], "num_lidar_points": 6, "distance": 22.499305863825455},
    {"instance_id": 232, "class": "yellow", "position": [18.030861474693026, 3.788611278009924, -1.5270677227808562], "box": [18.030861474693026, 3.788611278009924, -1.3670677227808563, 0.23, 0.23, 0.32, 0.0], "num_lidar_points": 14, "distance": 18.475237890054416},
    {"instance_id": 231, "class": "yellow", "position": [14.054113725066983, 3.1392670856550495, -1.5190448695610428], "box": [14.054113725066983, 3.1392670856550495, -1.3590448695610429, 0.23, 0.23, 0.32, 0.0], "num_lidar_points": 22, "distance": 14.464443072226233},
    {"instance_id": 229, "class": "yellow", "position": [6.039572530141982, 2.3460541960998853, -1.5028287423432687], "box": [6.039572530141982, 2.3460541960998853, -1.3428287423432688, 0.23, 0.23, 0.32, 0.0], "num_lidar_points": 0, "distance": 6.616917384186287},
    {"instance_id": 228, "class": "yellow", "position": [2.016685652288743, 2.2069339764998404, -1.4946959839060128], "box": [2.016685652288743, 2.2069339764998404, -1.3346959839060129, 0.23, 0.23, 0.32, 0.0], "num_lidar_points": 0, "distance": 3.273987166473245},
    {"instance_id": 199, "class": "blue", "position": [42.91863667779808, 6.396288540266141, -1.5772941291111806], "box": [42.91863667779808, 6.396288540266141, -1.4172941291111807, 0.23, 0.23, 0.32, 0.0], "num_lidar_points": 1, "distance": 43.41578749740222},
    {"instance_id": 198, "class": "blue", "position": [39.02003573832246, 4.995429593896461, -1.5694537642008584], "box": [39.02003573832246, 4.995429593896461, -1.4094537642008584, 0.23, 0.23, 0.32, 0.0], "num_lidar_points": 2, "distance": 39.363740495041355},
    {"instance_id": 196, "class": "blue", "position": [31.095175961069117, 2.373533554563892, -1.553508972352688], "box": [31.095175961069117, 2.373533554563892, -1.3935089723526881, 0.23, 0.23, 0.32, 0.0], "num_lidar_points": 5, "distance": 31.216750260725714},
    {"instance_id": 195, "class": "blue", "position": [22.97444217674149, 0.21320297435477187, -1.5371529805964315], "box": [22.97444217674149, 0.21320297435477187, -1.3771529805964315, 0.23, 0.23, 0.32, 0.0], "num_lidar_points": 8, "distance": 23.016667855549326},
    {"instance_id": 194, "class": "blue", "position": [18.847478407076892, -0.6366747353940809, -1.5288327939680357], "box": [18.847478407076892, -0.6366747353940809, -1.3688327939680358, 0.23, 0.23, 0.32, 0.0], "num_lidar_points": 17, "distance": 18.907842294713618},
    {"instance_id": 193, "class": "blue", "position": [14.685801392615303, -1.3161922498925378, -1.5204368561576018], "box": [14.685801392615303, -1.3161922498925378, -1.360436856157602, 0.23, 0.23, 0.32, 0.0], "num_lidar_points": 23, "distance": 14.807292562167154},
    {"instance_id": 192, "class": "blue", "position": [10.497871660037276, -1.8204178146984304, -1.51198212042641], "box": [10.497871660037276, -1.8204178146984304, -1.35198212042641, 0.23, 0.23, 0.32, 0.0], "num_lidar_points": 43, "distance": 10.739976073746064},
    {"instance_id": 191, "class": "blue", "position": [6.2916352860103615, -2.1468556776779195, -1.5035150962222872], "box": [6.2916352860103615, -2.1468556776779195, -1.3435150962222873, 0.23, 0.23, 0.32, 0.0], "num_lidar_points": 0, "distance": 6.782233900915375},
    {"instance_id": 190, "class": "blue", "position": [2.0731440329007427, -2.292691477506466, -1.494986898001116], "box": [2.0731440329007427, -2.292691477506466, -1.3349868980011161, 0.23, 0.23, 0.32, 0.0], "num_lidar_points": 0, "distance": 3.366979419304101}
  ]
}
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