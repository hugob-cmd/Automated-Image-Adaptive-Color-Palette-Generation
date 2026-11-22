import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import CSS4_COLORS, to_rgb
from PIL import Image
from skimage.color import rgb2lab
from sklearn.cluster import MeanShift

import segmentation_naive as naive
import segmentation_auto as auto
from kmeans_rapide import attribuer


# 1. Pixels de chaque mode (figure 5 de l'article)
def masques_des_modes(image, test=auto.est_unimodal_auto, largeur=naive.LARGEUR):
    H, S, I = naive.rgb_vers_hsi(image)
    H, S, I = H.ravel(), S.ravel(), I.ravel()
    Q = naive.Q

    classe_H = np.floor(H * Q / 360).astype(int) % Q
    colores = naive.pixels_colores(S)
    bornes_H, decalage = naive.ftc_circulaire(np.bincount(classe_H[colores], minlength=Q), test)
    groupe_H = naive.etiqueter((classe_H - decalage) % Q, bornes_H)
    ligne1 = [colores & (groupe_H == i) for i in range(len(bornes_H) - 1)]

    # Plus gros mode de teinte (en nombre de pixels colorés)
    i = np.argmax([m.sum() for m in ligne1])
    dans_i = groupe_H == i  # gris compris, comme dans l'algorithme
    classe_S = np.floor(S / largeur).astype(int)
    bornes_S = naive.ftc(np.bincount(classe_S[dans_i], minlength=classe_S.max() + 1), test)
    groupe_S = naive.etiqueter(classe_S, bornes_S)
    ligne2 = [dans_i & (groupe_S == j) for j in range(len(bornes_S) - 1)]

    j = np.argmax([m.sum() for m in ligne2])
    dans_ij = ligne2[j]
    classe_I = np.floor(I / largeur).astype(int)
    bornes_I = naive.ftc(np.bincount(classe_I[dans_ij], minlength=classe_I.max() + 1), test)
    groupe_I = naive.etiqueter(classe_I, bornes_I)
    ligne3 = [dans_ij & (groupe_I == k) for k in range(len(bornes_I) - 1)]

    # On enlève les modes vides
    return [[m for m in ligne if m.any()] for ligne in [ligne1, ligne2, ligne3]]


def afficher_pixels_des_modes(image, titre=""):
    # Comme la figure 5 : les pixels qui ne sont pas dans le mode sont mis en blanc
    lignes = masques_des_modes(image)
    noms = ["teinte", "saturation\n(plus gros mode de teinte)", "intensité\n(plus gros mode H, S)"]
    ncol = max(len(l) for l in lignes)
    fig, axes = plt.subplots(3, ncol, figsize=(2.4 * ncol, 2.4 * 3 * image.shape[0] / image.shape[1] + 0.6))
    axes = np.array(axes).reshape(3, ncol)
    for r, ligne in enumerate(lignes):
        for c in range(ncol):
            ax = axes[r, c]
            ax.set_xticks([])
            ax.set_yticks([])
            if c < len(ligne):
                vue = np.full_like(image, 255)
                masque = ligne[c].reshape(image.shape[:2])
                vue[masque] = image[masque]
                ax.imshow(vue)
                ax.set_title(f"{100 * masque.mean():.1f} % des pixels", fontsize=8)
            else:
                ax.axis("off")
        axes[r, 0].set_ylabel(noms[r], fontsize=8)
    fig.suptitle(titre)
    fig.tight_layout()
    return fig


# 2. Mean-Shift (comparaison de la section 6 de l'article)
def mean_shift(image, K_cible, espace="RGB", nb_echantillons=5000, graine=0):
    pixels = image.reshape(-1, 3).astype(np.float64)
    travail = rgb2lab(image / 255.0).reshape(-1, 3) if espace == "Lab" else pixels
    hasard = np.random.default_rng(graine)
    echantillon = travail[hasard.choice(len(travail), nb_echantillons, replace=False)]

    bas, haut = 1.0, 120.0
    meilleurs_centres = None
    for _ in range(12):
        largeur = (bas + haut) / 2
        centres = MeanShift(bandwidth=largeur, bin_seeding=True).fit(echantillon).cluster_centers_
        if meilleurs_centres is None or abs(len(centres) - K_cible) < abs(len(meilleurs_centres) - K_cible):
            meilleurs_centres = centres
        if len(centres) == K_cible:
            break
        if len(centres) > K_cible:
            bas = largeur
        else:
            haut = largeur

    labels = attribuer(travail, meilleurs_centres)
    # Couleur finale de chaque groupe = moyenne RGB de ses pixels
    palette = np.array([pixels[labels == k].mean(axis=0) if np.any(labels == k) else [0, 0, 0]
                        for k in range(len(meilleurs_centres))])
    finale = palette[labels].round().clip(0, 255).astype(np.uint8).reshape(image.shape)
    return finale, palette


# 3. Nommer les couleurs (section 5 de l'article)
def dictionnaire_nbs(chemin="donnees/NBS-ISCC-rgb.txt"):
    # Dictionnaire NBS/ISCC Centroids (267 noms), fichier de J. Jaffer
    noms, couleurs = [], []
    for ligne in open(chemin, encoding="latin-1"):
        if ligne.startswith("!") or not ligne.strip():
            continue
        r, g, b, nom = ligne.split()
        noms.append(nom)
        couleurs.append([int(r), int(g), int(b)])
    return noms, np.array(couleurs, dtype=np.float64)


def dictionnaire_x11():
    noms = sorted(CSS4_COLORS)
    couleurs = np.array([to_rgb(CSS4_COLORS[n]) for n in noms]) * 255
    return noms, couleurs


def nommer_palette(palette, effectifs, dictionnaire):
    noms, couleurs = dictionnaire
    lab_palette = rgb2lab(np.array(palette)[None, :, :] / 255)[0]
    lab_dico = rgb2lab(couleurs[None, :, :] / 255)[0]
    distances = ((lab_palette[:, None, :] - lab_dico[None, :, :]) ** 2).sum(axis=2)
    plus_proche = distances.argmin(axis=1)

    resultat = {}
    for k, d in enumerate(plus_proche):
        nom = noms[d]
        if nom not in resultat:
            resultat[nom] = {"pixels": 0, "couleur_dico": couleurs[d], "couleurs_palette": []}
        resultat[nom]["pixels"] += effectifs[k]
        resultat[nom]["couleurs_palette"].append(palette[k])
    return sorted(resultat.items(), key=lambda e: -e[1]["pixels"])


def effectifs_palette(finale, palette):
    # Nombre de pixels de l'image finale qui ont chaque couleur de la palette
    arrondie = np.array(palette).round().clip(0, 255).astype(np.uint8)
    pixels = finale.reshape(-1, 3)
    return np.array([np.all(pixels == c, axis=1).sum() for c in arrondie])


def afficher_noms(liste, titre, ax):
    # Une ligne par nom : couleur(s) de la palette | nom | couleur du dictionnaire
    ax.set_xlim(0, 10)
    ax.set_ylim(len(liste), 0)
    ax.axis("off")
    for r, (nom, infos) in enumerate(liste):
        for c, couleur in enumerate(infos["couleurs_palette"][:4]):
            ax.add_patch(plt.Rectangle((c * 0.5, r + 0.1), 0.45, 0.8, color=np.array(couleur) / 255))
        ax.text(2.2, r + 0.5, nom, va="center", fontsize=8)
        ax.add_patch(plt.Rectangle((8.6, r + 0.1), 1.2, 0.8, color=infos["couleur_dico"] / 255))
    ax.set_title(titre, fontsize=9)


if __name__ == "__main__":
    fleur = np.array(Image.open("images/fleur.tif").convert("RGB"))
    afficher_pixels_des_modes(fleur, "Pixels de chaque mode (fleur)")

    finale, palette, _, _, _ = auto.acopa_complet(fleur, "Lab")
    effectifs = effectifs_palette(finale, palette)
    fig, axes = plt.subplots(1, 2, figsize=(10, 6))
    afficher_noms(nommer_palette(palette, effectifs, dictionnaire_x11()), "Dictionnaire X11 (CSS4)", axes[0])
    afficher_noms(nommer_palette(palette, effectifs, dictionnaire_nbs()), "Dictionnaire NBS/ISCC", axes[1])

    for espace in ["RGB", "Lab"]:
        resultat, pal = mean_shift(fleur, len(palette), espace)
        print(f"Mean-Shift {espace} : {len(pal)} couleurs (ACoPa : {len(palette)})")
    plt.show()
