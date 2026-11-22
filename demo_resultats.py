import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import segmentation_naive as naive
import segmentation_auto as auto

IMAGES = ["images/Poivrons.jpeg", "images/fleur.tif", "images/astronaut.png",
          "images/coffee.png", "images/chelsea.png"]


def resultats_automatiques(image):
    # Renvoie une liste de (titre, image finale, palette)
    resultats = []

    graines, _, _, _ = naive.palette_acopa(image, naive.est_unimodal_naif, afficher=False)
    finale, palette = auto.acopa_kmeans(image, graines, "RGB")
    resultats.append((f"Naïf (SEUIL = {naive.SEUIL}) + K-means RGB", finale, palette))

    graines, _, _, _ = naive.palette_acopa(image, auto.est_unimodal_auto, afficher=False)
    for espace in ["RGB", "Lab"]:
        finale, palette = auto.acopa_kmeans(image, graines, espace)
        resultats.append((f"A contrario + K-means {espace}", finale, palette))
    return resultats


def dessiner_palette(ax, palette):
    # Une bande avec une case par couleur de la palette
    ax.imshow(np.array(palette)[None, :, :] / 255, aspect="auto")
    ax.set_xticks([])
    ax.set_yticks([])


def afficher_resultats(chemin, enregistrer=True):
    image = np.array(Image.open(chemin).convert("RGB"))
    resultats = resultats_automatiques(image)

    fig, axes = plt.subplots(2, 4, figsize=(14, 4.2), gridspec_kw={"height_ratios": [7, 1]})
    axes[0, 0].imshow(image)
    axes[0, 0].set_title("Original", fontsize=9)
    axes[1, 0].axis("off")
    for col, (titre, finale, palette) in enumerate(resultats, start=1):
        axes[0, col].imshow(finale)
        axes[0, col].set_title(f"{titre}\nK = {len(palette)}", fontsize=9)
        dessiner_palette(axes[1, col], palette)
        if enregistrer:
            nom = chemin.split("/")[-1].split(".")[0]
            suffixe = titre.split(" (")[0].split(" +")[0].replace(" ", "_").replace("ï", "i").lower()
            espace = titre.split()[-1]
            Image.fromarray(finale).save(f"resultats/{nom}_{suffixe}_{espace}.png")
    for ax in axes[0]:
        ax.axis("off")
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    for chemin in IMAGES:
        afficher_resultats(chemin)
    plt.show()
