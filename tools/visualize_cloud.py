import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

# 1. Carica il binario di CARLA (X, Y, Z, Intensity)
bin_file_path = "./../dataset/scenes/scene_0000/lidar/frame_000007.bin"
points = np.fromfile(bin_file_path, dtype=np.float32).reshape((-1, 4))[:, 0:3]

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

# Calcola le normali per dare l'effetto 3D di ombreggiatura e profondità
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=30))

# 2. Inizializza la finestra grafica avanzata di Open3D
gui.Application.instance.initialize()
window = gui.Application.instance.create_window("Visualizzatore Stile CARLA", 1024, 768)
scene_widget = gui.SceneWidget()
scene_widget.scene = rendering.Open3DScene(window.renderer)
window.add_child(scene_widget)

# 3. Configura il materiale per smussare i punti (Point Splatting)
material = rendering.MaterialRecord()
material.shader = "defaultLit"  # Attiva le luci e le ombre sui punti
material.base_color = [0.2, 0.6, 1.0, 1.0] # Colore azzurro stile LiDAR
material.point_size = 3.0       # Dimensione del punto aumentata
#material.point_size_is_relative = False # Mantiene la dimensione fissa in pixel

# 4. Aggiungi la geometria alla scena
scene_widget.scene.add_geometry("point_cloud", pcd, material)

# Regola la telecamera per inquadrare la scena
bounds = pcd.get_axis_aligned_bounding_box()
scene_widget.setup_camera(60, bounds, bounds.get_center())

# Esegui l'applicazione
gui.Application.instance.run()
