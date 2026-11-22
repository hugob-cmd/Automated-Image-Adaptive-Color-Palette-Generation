import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio

# On réutilise la conversion HSI et le cylindre des gris écrits dans segmentation_naive.py
from segmentation_naive import rgb_vers_hsi as rgb_to_hsi, pixels_colores as chromatic_mask
from kmeans_rapide import kmeans, centres_aleatoires, image_depuis_labels

# Version simplifiée d'ACoPa (article "Automatic color palette", Delon et al.)
# Idée de l'article : au lieu de choisir K à la main, on regarde les "bosses" (modes) de l'histogramme de teinte
# Une bosse = un groupe de couleurs de teinte proche (le rouge des poivrons, le vert, ...)
# Nombre de bosses = K, et la couleur moyenne de chaque bosse = centre de départ de K-means
# Simplification par rapport à l'article : on ne fait pas le test statistique "a contrario" (PAV + seuil)
# on garde juste les pics bien séparés de l'histogramme lissé, et on ne redécoupe pas en saturation

IMAGE = "images/Poivrons.jpeg"
Q = 360           # nombre de classes de l'histogramme de teinte (1 classe = 1 degré)
LISSAGE = 9       # largeur de la moyenne glissante (en classes) pour enlever le bruit de l'histogramme
CREUX = 0.6       # deux bosses sont séparées si le creux entre elles descend sous 60 % de la plus petite
TAILLE_MIN = 0.01 # une bosse qui contient moins de 1 % des pixels est ignorée


def lisser(h, largeur, circulaire):
    noyau = np.ones(largeur) / largeur
    m = largeur // 2
    if circulaire:
        h_etendu = np.concatenate([h[-m:], h, h[:m]])
    else:
        h_etendu = np.concatenate([np.zeros(m), h, np.zeros(m)])
    return np.convolve(h_etendu, noyau, mode="valid")


def trouver_bosses(h, circulaire):
    # 1. On cherche tous les minimums locaux : ce sont les frontières possibles entre deux bosses
    n = len(h)
    minimums = []
    for i in range(n):
        gauche = h[(i - 1) % n] if (circulaire or i > 0) else np.inf
        droite = h[(i + 1) % n] if (circulaire or i < n - 1) else np.inf
        if h[i] <= gauche and h[i] < droite:
            minimums.append(i)
    if not circulaire:
        # Pour un histogramme non circulaire, les deux bords sont aussi des frontières
        minimums = sorted(set([0] + minimums + [n]))
    if len(minimums) == 0:
        return [(0, n)]  # pas de creux : une seule bosse qui fait tout le tour

    # 2. On fusionne les bosses voisines tant que le creux qui les sépare n'est pas assez profond
    # (sinon un petit creux dû au bruit couperait une vraie bosse en deux)
    fusion = True
    while fusion and len(minimums) > 1:
        fusion = False
        for j in range(len(minimums)):
            # Bosse j : de minimums[j] à minimums[j+1], bosse j+1 : de minimums[j+1] à minimums[j+2]
            if not circulaire and j + 2 >= len(minimums):
                break
            a, b, c = minimums[j], minimums[(j + 1) % len(minimums)], minimums[(j + 2) % len(minimums)]
            pic1 = max_entre(h, a, b)
            pic2 = max_entre(h, b, c)
            if h[b % n] > CREUX * min(pic1, pic2):
                # Creux pas assez profond : on enlève la frontière b
                minimums.pop((j + 1) % len(minimums))
                fusion = True
                break

    # 3. On renvoie la liste des intervalles (début, fin) de chaque bosse
    if circulaire:
        return [(minimums[j], minimums[(j + 1) % len(minimums)]) for j in range(len(minimums))]
    return [(minimums[j], minimums[j + 1]) for j in range(len(minimums) - 1)]


def max_entre(h, debut, fin):
    # Plus grande valeur de h entre debut et fin, en faisant le tour si fin < debut (cas circulaire)
    if fin > debut:
        return h[debut:fin + 1].max() if fin < len(h) else h[debut:].max()
    return max(h[debut:].max(), h[:fin + 1].max())


def dans_intervalle(valeurs, debut, fin):
    # Vrai pour les valeurs entre debut et fin (avec le tour du cercle si fin < debut)
    if fin > debut:
        return (valeurs >= debut) & (valeurs < fin)
    return (valeurs >= debut) | (valeurs < fin)


def palette_acopa(pixels):
    liste_pixels = pixels.reshape(-1, 3).astype(np.float64)
    H, S, I = rgb_to_hsi(pixels)
    H, S, I = H.ravel(), S.ravel(), I.ravel()

    # 1. On sépare les pixels colorés des pixels gris (fonction de Segmentation.py)
    # Pour un pixel gris la teinte ne veut rien dire, donc on les traite à part avec l'intensité
    colores = chromatic_mask(S, Q)
    print(f"{colores.sum()} pixels colorés, {(~colores).sum()} pixels gris")

    centres = []

    # 2. Pixels colorés : bosses de l'histogramme de teinte (circulaire)
    h_teinte, _ = np.histogram(H[colores], bins=Q, range=(0, 360))
    h_teinte = lisser(h_teinte, LISSAGE, circulaire=True)
    for debut, fin in trouver_bosses(h_teinte, circulaire=True):
        dedans = colores & dans_intervalle(H, debut, fin)
        if dedans.sum() > TAILLE_MIN * len(liste_pixels):
            print(f"   bosse de teinte entre {debut} et {fin} degrés : {dedans.sum()} pixels")
            centres.append(liste_pixels[dedans].mean(axis=0))

    # 3. Pixels gris : même chose avec l'histogramme d'intensité (pas circulaire : noir -> blanc)
    h_intensite, _ = np.histogram(I[~colores], bins=256, range=(0, 256))
    h_intensite = lisser(h_intensite, LISSAGE, circulaire=False)
    for debut, fin in trouver_bosses(h_intensite, circulaire=False):
        dedans = (~colores) & (I >= debut) & (I < fin)
        if dedans.sum() > TAILLE_MIN * len(liste_pixels):
            print(f"   bosse de gris entre {debut} et {fin} d'intensité : {dedans.sum()} pixels")
            centres.append(liste_pixels[dedans].mean(axis=0))

    return np.array(centres)


if __name__ == "__main__":
    image = Image.open(IMAGE).convert("RGB")
    image.show()
    pixels = np.array(image)

    # On laisse l'histogramme choisir K
    centres_depart = palette_acopa(pixels)
    K = len(centres_depart)
    print(f"Nombre de couleurs trouvé automatiquement : K = {K}")
    print(centres_depart.round())

    # K-means initialisé avec les couleurs moyennes des bosses
    centres, labels = kmeans(pixels, centres_depart, afficher=False)
    resultat = image_depuis_labels(centres, labels)
    image_finale = Image.fromarray(resultat)
    image_finale.show()
    image_finale.save(f"resultats/Poivrons_acopa_simple_k{K}.png")

    # Pour comparer : K-means avec le même K mais des centres au hasard
    centres_h, labels_h = kmeans(pixels, centres_aleatoires(pixels, K), afficher=False)
    resultat_h = image_depuis_labels(centres_h, labels_h)
    print(f"PSNR ACoPa simplifié       : {peak_signal_noise_ratio(pixels, resultat, data_range=255):.2f} dB")
    print(f"PSNR K-means même K, hasard : {peak_signal_noise_ratio(pixels, resultat_h, data_range=255):.2f} dB")
    # Remarque : le PSNR peut être proche (ou même moins bon), car ACoPa ne cherche pas à minimiser l'erreur
    # mais à garder les couleurs qui ont un sens pour l'oeil, même celles des petites zones
