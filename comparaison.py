# --------------------------------------------------------------------------
# Projet 4IM01 - Palette automatique de couleurs
# Auteur : Hugo (binôme) - comparaison chiffrée des méthodes classiques.
# Modifications (Claude, 21/11) : "median cut.py" renommé median_cut.py (import normal),
# chemins images/ et resultats/.
# --------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio

from popularite import popularite, remplacer_par_palette
from kmeans_rapide import centres_aleatoires, kmeans, image_depuis_labels

# Comparaison chiffrée des méthodes : pour chaque K, on mesure l'erreur entre l'image d'origine et l'image quantifiée
# MSE = moyenne des (pixel d'origine - pixel quantifié)² -> plus c'est petit, mieux c'est
# PSNR = 10 * log10(255² / MSE), en décibels -> plus c'est grand, mieux c'est
# (au-dessus de ~30 dB on voit difficilement la différence à l'oeil)

IMAGE = "images/Poivrons.jpeg"
VALEURS_K = [2, 4, 8, 12, 16, 32, 64]

import median_cut as median_cut_module


def methode_popularite(pixels, K):
    palette = popularite(pixels, K)
    return remplacer_par_palette(pixels, palette)


def methode_median_cut(pixels, K):
    palette = median_cut_module.median_cut(pixels, K)
    return median_cut_module.remap(pixels, palette)


def methode_kmeans(pixels, K):
    # K-means avec des centres de départ tirés au hasard dans l'image
    centres, labels = kmeans(pixels, centres_aleatoires(pixels, K), afficher=False)
    return image_depuis_labels(centres, labels)


def methode_median_cut_puis_kmeans(pixels, K):
    # Idée : la palette du median cut sert de centres de départ pour K-means
    # (même principe que ACoPa : une bonne initialisation, puis K-means pour affiner)
    palette = median_cut_module.median_cut(pixels, K)
    centres, labels = kmeans(pixels, palette, afficher=False)
    return image_depuis_labels(centres, labels)


METHODES = {
    "Popularité": methode_popularite,
    "Median cut": methode_median_cut,
    "K-means (init. au hasard)": methode_kmeans,
    "Median cut + K-means": methode_median_cut_puis_kmeans,
}


if __name__ == "__main__":
    pixels = np.array(Image.open(IMAGE).convert("RGB"))

    # resultats["Popularité"] = liste des PSNR pour chaque K, etc.
    resultats_mse = {nom: [] for nom in METHODES}
    resultats_psnr = {nom: [] for nom in METHODES}

    for K in VALEURS_K:
        print(f"K = {K}")
        for nom, methode in METHODES.items():
            new_pixels = methode(pixels, K)
            mse = mean_squared_error(pixels, new_pixels)
            psnr = peak_signal_noise_ratio(pixels, new_pixels, data_range=255)
            resultats_mse[nom].append(mse)
            resultats_psnr[nom].append(psnr)
            print(f"   {nom:28s} MSE = {mse:8.1f}   PSNR = {psnr:5.2f} dB")
            # On garde l'image pour K = 8 pour pouvoir comparer à l'oeil
            if K == 8:
                Image.fromarray(new_pixels).save(f"resultats/comparaison_k8_{nom.replace(' ', '_').replace('.', '')}.png")

    # Courbes : PSNR en fonction de K pour chaque méthode
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for nom in METHODES:
        ax1.plot(VALEURS_K, resultats_mse[nom], marker="o", label=nom)
        ax2.plot(VALEURS_K, resultats_psnr[nom], marker="o", label=nom)
    # Échelle log en x car on double K à chaque fois
    for ax in (ax1, ax2):
        ax.set_xscale("log", base=2)
        ax.set_xticks(VALEURS_K)
        ax.set_xticklabels(VALEURS_K)
        ax.set_xlabel("Nombre de couleurs K")
        ax.grid(alpha=0.3)
    ax1.set_ylabel("MSE (plus bas = mieux)")
    ax2.set_ylabel("PSNR en dB (plus haut = mieux)")
    ax2.legend()
    fig.suptitle(f"Erreur de quantification selon K ({IMAGE})")
    fig.tight_layout()
    fig.savefig("resultats/comparaison_mse_psnr.png", dpi=130)
    plt.show()
