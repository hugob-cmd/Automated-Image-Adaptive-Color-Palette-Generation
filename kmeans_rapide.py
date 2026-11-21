# --------------------------------------------------------------------------
# Projet 4IM01 - Palette automatique de couleurs
# Auteur : Hugo (binôme) - K-means vectorisé avec NumPy.
# Modifications (Claude, 21/11) : ajout de centres_aleatoires, centre_le_plus_proche,
# image_depuis_labels et du paramètre "afficher", qui étaient importés par
# popularite.py, comparaison.py et acopa_simple.py mais n'existaient pas ici
# (ces trois scripts plantaient à l'import). kmeans accepte aussi une image (h, w, 3).
# --------------------------------------------------------------------------
import sys
import time
import numpy as np
from PIL import Image

# Même algorithme que k-means-iterated.py, mais sans boucles sur les pixels :
# NumPy traite tous les pixels d'un coup, on ne boucle que sur les K centres.

IMAGE_PATH = "images/Poivrons.jpeg"
K = 12
SEUIL = 0.5      # on s'arrête quand aucun centre ne bouge de plus de 0,5
MAX_ITER = 100   # sécurité si ça ne converge pas


def centres_initiaux(pixels, k, graine=0):
    # K pixels de l'image tirés au hasard (graine fixée pour avoir toujours le même résultat)
    rng = np.random.default_rng(graine)
    indices = rng.choice(len(pixels), size=k, replace=False)
    return pixels[indices].copy()


def centres_aleatoires(pixels, k, graine=0):
    # Même chose que centres_initiaux, mais accepte aussi une image (h, w, 3)
    return centres_initiaux(pixels.reshape(-1, 3), k, graine)


def attribuer(pixels, centres):
    # Tableau (nb_pixels, K) : distance au carré de chaque pixel à chaque centre
    distances = np.zeros((len(pixels), len(centres)))
    for k in range(len(centres)):
        distances[:, k] = ((pixels - centres[k]) ** 2).sum(axis=1)
    # Pour chaque pixel, numéro du centre le plus proche
    return distances.argmin(axis=1)


def centre_le_plus_proche(pixels, centres):
    # Autre nom pour attribuer (utilisé par popularite.py)
    return attribuer(pixels, centres)


def kmeans(pixels, centres, seuil=SEUIL, max_iter=MAX_ITER, afficher=True):
    # pixels : tableau (nb_pixels, 3) ou image (h, w, 3), centres : tableau (K, 3)
    # Si on donne une image, les labels sont rendus sous forme d'image (h, w)
    forme = pixels.shape[:-1]
    pixels = pixels.reshape(-1, 3).astype(np.float64)
    centres = centres.astype(np.float64)

    for iteration in range(1, max_iter + 1):
        # 1. Chaque pixel va vers son centre le plus proche
        labels = attribuer(pixels, centres)

        # 2. Chaque centre devient la moyenne de ses pixels
        nouveaux_centres = centres.copy()
        for k in range(len(centres)):
            groupe = pixels[labels == k]
            if len(groupe) > 0:
                nouveaux_centres[k] = groupe.mean(axis=0)
            else:
                if afficher:
                    print(f"Groupe {k} n'a aucun pixel associé")

        # 3. De combien le centre qui a le plus bougé s'est-il déplacé ?
        deplacement = np.linalg.norm(nouveaux_centres - centres, axis=1).max()
        centres = nouveaux_centres
        if afficher:
            print(f"iteration {iteration} : déplacement max = {deplacement:.2f}")

        # 4. Arrêt quand les centres ne bougent presque plus
        if deplacement < seuil:
            break

    # Labels recalculés avec les centres finaux
    return centres, attribuer(pixels, centres).reshape(forme)


def image_depuis_labels(centres, labels):
    # Chaque pixel prend la couleur de son centre -> image (h, w, 3) en uint8
    return centres[labels].round().clip(0, 255).astype(np.uint8)


def quantifier(image, k):
    # Image (h, w, 3) -> image avec k couleurs, et la palette
    h, w, _ = image.shape
    pixels = image.reshape(-1, 3)
    centres, labels = kmeans(pixels, centres_initiaux(pixels, k))
    new_pixels = centres[labels].reshape(h, w, 3)
    return new_pixels.round().astype(np.uint8), centres


if __name__ == "__main__":
    # 1. Lecture de l'image (chemin en argument, sinon IMAGE_PATH)
    path = sys.argv[1] if len(sys.argv) > 1 else IMAGE_PATH
    image = np.array(Image.open(path).convert("RGB"))

    # 2. Quantification et temps de calcul
    debut = time.time()
    resultat, palette = quantifier(image, K)
    print(f"K = {K} : {time.time() - debut:.2f} s")
    print("Palette :")
    print(palette.round().astype(int))

    # 3. Enregistrement et affichage
    Image.fromarray(resultat).save("resultats/kmeans_rapide_resultat.png")
    Image.fromarray(resultat).show()


