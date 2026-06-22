
#proiezione dei coni 3d (frame world carla) nelle immagini 2D delle camere

import numpy as np


def _world_to_camera_matrix(cam_world_transform):
    
    #restituisce la matrice 4x4 world->camera (l'inversa della trasformazione della camera nel mondo)
    
    return np.array(cam_world_transform.get_inverse_matrix(), dtype=np.float64)


def project_cone_to_image(
    cone_world_xyz,
    world_to_cam,
    K,
    image_width,
    image_height,
):
    
    #proietta un singolo cono dal frame world carla al pixel (u, v) dell'immagine

    #punto omogeneo nel frame world
    point_world = np.array([
        float(cone_world_xyz[0]),
        float(cone_world_xyz[1]),
        float(cone_world_xyz[2]),
        1.0,
    ])

    #world -> camera 
    point_cam_ue = world_to_cam @ point_world

    #UE -> frame camera standard: (x, y, z) -> (y, -z, x)
    #così z_std diventa la profondità lungo l'asse ottico (forward)
    point_cam = np.array([
        point_cam_ue[1],
        -point_cam_ue[2],
        point_cam_ue[0],
    ])

    depth = point_cam[2]  #z standard = profondità ottica

    #coni dietro la camera: depth <= 0 -> non proiettabili
    if depth <= 1e-6:
        return {
            "u": None, "v": None, "depth_m": float(depth),
            "in_front": False, "in_image": False,
        }

    #proiezione pinhole: [u, v, w]^T = K @ point_cam  poi normalizza per w
    uvw = K @ point_cam
    u = uvw[0] / uvw[2]
    v = uvw[1] / uvw[2]

    in_image = (0.0 <= u < image_width) and (0.0 <= v < image_height)

    return {
        "u": float(u),
        "v": float(v),
        "depth_m": float(depth),
        "in_front": True,
        "in_image": bool(in_image),
    }


def project_cones_for_camera(
    cone_annotations_world,
    cam_world_transform,
    K,
    image_width,
    image_height,
):
    
    #proietta una lista di coni nell'immagine di una camera
    world_to_cam = _world_to_camera_matrix(cam_world_transform)

    out = []
    for ann in cone_annotations_world:
        proj = project_cone_to_image(
            ann["world_xyz"],
            world_to_cam,
            K,
            image_width,
            image_height,
        )
        if not proj["in_front"] or not proj["in_image"]:
            continue

        out.append({
            "color": ann["class"],
            "center_px": [proj["u"], proj["v"]],
            "depth_m": proj["depth_m"],
            "fully_in_image": True,
            "instance_id": int(ann.get("instance_id", -1)),
        })
    return out