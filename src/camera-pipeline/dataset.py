


def gaussian_2d(shape: tuple[int, int], center: tuple[float, float], sigma: float) -> np.ndarray:
    
    #genera una gaussiana 2d con centro su center, con deviazione sigma e dimensione shape

    H, W = shape
    cx, cy = center
    y = np.arange(H, dtype=np.float32)[:, None]
    x = np.arange(W, dtype=np.float32)[None, :]

    gauss = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma ** 2))
    return gauss


def generate_heatmap_offset_mask(
    cones: List[Dict],
    image_size: Tuple[int, int],
    stride: int,
    num_classes: int,
    sigma: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    #genera heatmap gaussiana, offset e offset mask per il calcolo della loss

    #heatmap: gaaussiana in corrispondenza dei centri dei coni
    #offset: vettore di offset per recuperare precisione sul centro esatto del cono
    #offset_mask: maschera per calcolare la loss solo in corrispondenza dei centri dei coni

    H, W = image_size
    H_feat, W_feat = H // stride, W // stride

    heatmap = np.zeros((num_classes, H_feat, W_feat), dtype=np.float32)
    offset = np.zeros((2, H_feat, W_feat), dtype=np.float32)
    offset_mask = np.zeros((H_feat, W_feat), dtype=np.float32)

    color_to_class = {
        "red": 0,
        "blue": 1,
    }

    for cone in cones:
        if not cone.get("fully_in_image", True):
            continue

        color = cone["color"]
        if color not in color_to_class:
            continue

        class_idx = color_to_class[color]

        #centro del cono in coordinate immagine quindi pixel
        cx_img, cy_img = cone["center_px"]

        #coordinate del centro del cono in coordinate feature map quindi divido per stride
        cx_feat = cx_img / stride
        cy_feat = cy_img / stride

        #pixel intero più vicino al centro del cono in coordinate feature map
        cx_feat_int = int(round(cx_feat))
        cy_feat_int = int(round(cy_feat))

        if not (0 <= cx_feat_int < W_feat and 0 <= cy_feat_int < H_feat):
            continue

        #genera gaussiana 2d centrata sul cono
        gauss = gaussian_2d((H_feat, W_feat), (cx_feat, cy_feat), sigma)

        heatmap[class_idx] = np.maximum(heatmap[class_idx], gauss)

        #crea offset e offset mask
        offset[0, cy_feat_int, cx_feat_int] = cx_feat - cx_feat_int
        offset[1, cy_feat_int, cx_feat_int] = cy_feat - cy_feat_int
        offset_mask[cy_feat_int, cx_feat_int] = 1.0





