# --------------------------------------------------------------------------
# Projet 4IM01 - Palette automatique de couleurs
# Auteur : Hugo (binôme) - algorithme de popularité.
# Modification (Claude, 21/11) : chemin de l'image (images/), plus d'autre changement.
# --------------------------------------------------------------------------
import numpy as np
from PIL import Image
from kmeans_rapide import centre_le_plus_proche

# Algorithme de popularité : on garde les K couleurs qui apparaissent le plus souvent dans l'image
# Problème : une image a souvent des dizaines de milliers de couleurs différentes,
# presque toutes présentes une seule fois, donc compter les couleurs exactes ne marche pas bien
# Solution : on regroupe d'abord les couleurs proches en "cases" (on enlève les bits de poids faible)

K = 12
BITS = 4


def popularite(pixels, K, bits=BITS):
    # 1. On met les pixels dans un tableau (nb_pixels, 3) et on les convertit en int64 pour éviter les débordements
    liste_pixels = pixels.reshape(-1, 3).astype(np.int64)

    # 2. On enlève les bits de poids faible pour regrouper les couleurs proches dans des "cases"
    decalage = 8 - bits
    cases = liste_pixels >> decalage

    # 3. On donne un numéro unique à chaque case (comme écrire un nombre en base 2^bits)
    n = 2 ** bits
    numeros = cases[:, 0] * n * n + cases[:, 1] * n + cases[:, 2]

    # 4. On compte combien de pixels tombent dans chaque case
    compte = np.bincount(numeros, minlength=n ** 3)

    # 5. On garde les K cases les plus remplies (argsort trie du plus petit au plus grand, d'où le [::-1])
    meilleures_cases = np.argsort(compte)[::-1][:K]

    # 6. La couleur de chaque case = la moyenne des vrais pixels qui sont dedans
    # (mieux que de prendre le centre de la case, qui peut ne correspondre à aucun pixel)
    palette = []
    for c in meilleures_cases:
        palette.append(liste_pixels[numeros == c].mean(axis=0))
    return np.array(palette)


def remplacer_par_palette(pixels, palette):
    # Chaque pixel prend la couleur de la palette la plus proche (distance euclidienne dans RGB)
    # Même idée que la double boucle de k-means.py, mais calculée d'un coup avec NumPy
    liste_pixels = pixels.reshape(-1, 3).astype(np.float64)
    labels = centre_le_plus_proche(liste_pixels, palette)
    new_pixels = palette[labels].reshape(pixels.shape)
    return new_pixels.round().astype(np.uint8)


if __name__ == "__main__":
    image = Image.open("images/Poivrons.jpeg").convert("RGB")
    image.show()
    pixels = np.array(image)

    palette = popularite(pixels, K)
    print("Palette trouvée :")
    print(palette.round())

    new_pixels = remplacer_par_palette(pixels, palette)
    image_finale = Image.fromarray(new_pixels)
    image_finale.show()
    image_finale.save(f"resultats/Poivrons_popularite_k{K}.png")

    # Remarque : les petites zones de couleur (un petit objet rouge par ex.) disparaissent
    # car leurs cases ne sont jamais dans les K plus remplies -> c'est le défaut connu de cette méthode
