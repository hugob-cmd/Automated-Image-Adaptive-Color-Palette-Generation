"""Median cut (Heckbert, 1982) : quantification de couleurs avec NumPy.

Usage en ligne de commande :
    python median_cut.py entree.png sortie.png 16
Sans argument, un test sur une image synthétique est exécuté.
"""
import sys
import numpy as np


def median_cut(pixels: np.ndarray, k: int) -> np.ndarray:
    """Calcule une palette de k couleurs.

    pixels : tableau (n, 3) de uint8/float (RGB)
    retourne : tableau (m, 3) de float, m <= k
    """
    pixels = np.asarray(pixels, dtype=np.float64).reshape(-1, 3)
    boxes = [pixels]

    while len(boxes) < k:
        # 1. Choix de la boîte : effectif x étendue maximale (variante courante)
        scores = [
            len(b) * (b.max(axis=0) - b.min(axis=0)).max() if len(b) > 1 else -1
            for b in boxes
        ]
        i = int(np.argmax(scores))
        if scores[i] <= 0:  # plus rien à couper (boîtes homogènes)
            break
        box = boxes.pop(i)

        # 2. Axe d'étendue maximale
        axis = int(np.argmax(box.max(axis=0) - box.min(axis=0)))

        # 3. Coupe à la médiane (np.argpartition : O(n), sans tri complet)
        mid = len(box) // 2
        order = np.argpartition(box[:, axis], mid)
        boxes.append(box[order[:mid]])
        boxes.append(box[order[mid:]])

    # 4. Couleur moyenne de chaque boîte
    return np.array([b.mean(axis=0) for b in boxes])


def remap(image: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """Remplace chaque pixel par la couleur de palette la plus proche."""
    h, w, _ = image.shape
    flat = image.reshape(-1, 3).astype(np.float64)
    out = np.empty_like(flat)
    # Par blocs pour limiter la mémoire : (bloc, k, 3)
    for s in range(0, len(flat), 65536):
        chunk = flat[s:s + 65536]
        d = ((chunk[:, None, :] - palette[None, :, :]) ** 2).sum(axis=2)
        out[s:s + 65536] = palette[d.argmin(axis=1)]
    return out.reshape(h, w, 3).round().astype(np.uint8)


def quantize(image: np.ndarray, k: int = 16):
    palette = median_cut(image.reshape(-1, 3), k)
    return remap(image, palette), palette


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(((a.astype(np.float64) - b.astype(np.float64)) ** 2).mean())


def _synthetic_image(h=200, w=200, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x, y = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    img = np.stack([x, y, 1 - x * y], axis=2) * 255
    img += rng.normal(0, 8, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def run_on_image(path: str, k_values) -> None:
    """Quantifie l'image pour chaque k et sauvegarde nom_k<k>.png à côté."""
    import os
    from PIL import Image
    img = np.array(Image.open(path).convert("RGB"))
    h, w, _ = img.shape
    print(f"Image : {path} ({w}x{h}, {len(np.unique(img.reshape(-1, 3), axis=0))} couleurs distinctes)")
    base, _ = os.path.splitext(path)
    for k in k_values:
        res, pal = quantize(img, k)
        out = f"{base}_k{k}.png"
        Image.fromarray(res).save(out)
        print(f"k={k:3d} -> MSE = {mse(img, res):8.1f}  ({out})")


# ============================================================
# >>> COLLE ICI LE CHEMIN DE TON IMAGE <<<
IMAGE_PATH = r"C:\Users\azizd\Downloads\poivron.jpg"
K_VALUES = (4,12)
# ============================================================


if __name__ == "__main__":
    if len(sys.argv) == 4:
        from PIL import Image
        src, dst, k = sys.argv[1], sys.argv[2], int(sys.argv[3])
        img = np.array(Image.open(src).convert("RGB"))
        res, pal = quantize(img, k)
        Image.fromarray(res).save(dst)
        print(f"{len(pal)} couleurs, MSE = {mse(img, res):.1f}")
    elif IMAGE_PATH:
        run_on_image(IMAGE_PATH, K_VALUES)
    else:
        img = _synthetic_image()
        for k in (2, 4, 16, 64, 256):
            res, pal = quantize(img, k)
            n_unique = len(np.unique(res.reshape(-1, 3), axis=0))
            print(f"k={k:3d} -> palette {len(pal):3d}, couleurs réelles {n_unique:3d}, MSE = {mse(img, res):8.1f}")