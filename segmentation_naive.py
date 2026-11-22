import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

IMAGE = "images/Poivrons.jpeg"
Q = 360        # nombre de classes de l'histogramme de teinte (1 classe = 1 degré)
SEUIL = 0.05   # écart maximal toléré (en fraction des pixels du morceau) pour dire "unimodal"
LARGEUR = 4    # largeur des classes pour la saturation et l'intensité


# 1. Passage de RGB à HSI
def rgb_vers_hsi(image):
    rgb = image.astype(np.float64)
    R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    I = (R + G + B) / 3
    S = np.sqrt((R - I) ** 2 + (G - I) ** 2 + (B - I) ** 2)
    H = np.degrees(np.arctan2(np.sqrt(3) * (G - B), 2 * R - G - B)) % 360
    return H, S, I


def pixels_colores(S, Q=Q):
    return S > Q / (2 * np.pi)


# 2. Pool Adjacent Violators (estimateur de Grenander)
def pav_decroissant(h):
    valeurs = []
    tailles = []
    for x in h:
        valeurs.append(float(x))
        tailles.append(1)
        while len(valeurs) > 1 and valeurs[-2] < valeurs[-1]:
            t = tailles[-2] + tailles[-1]
            v = (valeurs[-2] * tailles[-2] + valeurs[-1] * tailles[-1]) / t
            valeurs.pop()
            tailles.pop()
            valeurs[-1] = v
            tailles[-1] = t
    return np.repeat(valeurs, tailles)


def pav_croissant(h):
    # Même chose en croissant : on retourne h, on applique pav_decroissant, on retourne
    return pav_decroissant(np.asarray(h)[::-1])[::-1]


def ajustement_unimodal(h, c):
    return np.concatenate([pav_croissant(h[:c]), pav_decroissant(h[c:])])


def pics_candidats(h, nb=3):
    return np.argsort(h)[::-1][:nb]


# 3. Test d'unimodalité NAÏF (seuil réglé à la main)
def ecart(h, ajustement):
    N = h.sum()
    if N == 0:
        return 0
    return np.abs(h - ajustement).sum() / (2 * N)


def est_unimodal_naif(h, seuil=SEUIL):
    h = np.asarray(h, dtype=np.float64)
    if len(h) < 3 or h.sum() == 0:
        return True
    for c in pics_candidats(h):
        if ecart(h, ajustement_unimodal(h, c)) <= seuil:
            return True
    return False


# 4. Algorithme FTC (Fine To Coarse) de l'article
def minima_locaux(h):
    # Indices des minima locaux de h (on prend le début et la fin des plateaux)
    m = []
    for i in range(1, len(h) - 1):
        if (h[i] < h[i - 1] and h[i] <= h[i + 1]) or (h[i] <= h[i - 1] and h[i] < h[i + 1]):
            m.append(i)
    return m


def ftc(h, est_unimodal, graine=0):
    h = np.asarray(h, dtype=np.float64)
    L = len(h)
    bornes = [0] + minima_locaux(h) + [L]
    hasard = np.random.default_rng(graine)

    j = 2
    while j <= len(bornes) - 1:  # len(bornes) - 1 = nombre de morceaux
        fusion = False
        debuts = np.arange(len(bornes) - j)  # union des morceaux k, k+1, ..., k+j-1
        hasard.shuffle(debuts)
        for k in debuts:
            a, b = bornes[k], bornes[k + j]
            if est_unimodal(h[a:b]):
                bornes = bornes[:k + 1] + bornes[k + j:]  # on enlève les bornes intérieures
                fusion = True
                break
        if fusion:
            j = 2
        else:
            j += 1
    return bornes


def etiqueter(classes, bornes):
    # Numéro du morceau de chaque valeur : classes dans [bornes[k], bornes[k+1][ -> k
    return np.searchsorted(bornes, classes, side="right") - 1


def ftc_circulaire(h, est_unimodal):
    decalage = int(np.argmin(h))
    bornes = ftc(np.roll(h, -decalage), est_unimodal)
    return bornes, decalage


# 5. Palette hiérarchique : teinte -> saturation -> intensité (algorithme ACoPa)
def palette_acopa(image, est_unimodal, afficher=True, largeur=LARGEUR):
    pixels = image.reshape(-1, 3).astype(np.float64)
    H, S, I = rgb_vers_hsi(image)
    H, S, I = H.ravel(), S.ravel(), I.ravel()

    classe_H = np.floor(H * Q / 360).astype(int) % Q
    NB_S = int(210 // largeur) + 1
    NB_I = int(256 // largeur) + 1
    classe_S = np.floor(S / largeur).astype(int)
    classe_I = np.floor(I / largeur).astype(int)

    # Étape 1 : FTC sur l'histogramme de teinte des pixels colorés
    colores = pixels_colores(S)
    h_teinte = np.bincount(classe_H[colores], minlength=Q)
    bornes_H, decalage = ftc_circulaire(h_teinte, est_unimodal)

    # Étape 2 : TOUS les pixels (gris compris) sont rattachés à un intervalle de teinte
    groupe_H = etiqueter((classe_H - decalage) % Q, bornes_H)

    graines, effectifs = [], []
    niveaux = {"teinte": [], "saturation": [], "intensite": []}
    for i in range(len(bornes_H) - 1):
        dans_i = groupe_H == i
        if dans_i.sum() == 0:
            continue
        niveaux["teinte"].append(pixels[dans_i].mean(axis=0))

        # Étape 3 : FTC sur l'histogramme de saturation des pixels de ce mode de teinte
        h_sat = np.bincount(classe_S[dans_i], minlength=NB_S)
        bornes_S = ftc(h_sat, est_unimodal)
        groupe_S = etiqueter(classe_S, bornes_S)

        for j in range(len(bornes_S) - 1):
            dans_ij = dans_i & (groupe_S == j)
            if dans_ij.sum() == 0:
                continue
            niveaux["saturation"].append(pixels[dans_ij].mean(axis=0))

            # Étape 4 : FTC sur l'histogramme d'intensité
            h_int = np.bincount(classe_I[dans_ij], minlength=NB_I)
            bornes_I = ftc(h_int, est_unimodal)
            groupe_I = etiqueter(classe_I, bornes_I)

            for k in range(len(bornes_I) - 1):
                dans_ijk = dans_ij & (groupe_I == k)
                if dans_ijk.sum() == 0:
                    continue
                couleur = pixels[dans_ijk].mean(axis=0)
                niveaux["intensite"].append(couleur)
                graines.append(couleur)
                effectifs.append(int(dans_ijk.sum()))

    if afficher:
        print(f"{len(bornes_H) - 1} modes de teinte, {len(niveaux['saturation'])} après la saturation, "
              f"{len(graines)} après l'intensité")
    infos = {"h_teinte": h_teinte, "bornes_H": bornes_H, "decalage": decalage}
    return np.array(graines), np.array(effectifs), niveaux, infos


# 6. Affichages
def afficher_segmentation_teinte(h_teinte, bornes, decalage, titre=""):
    # Histogramme de teinte en noir + bornes trouvées par le FTC en rouge
    plt.figure(figsize=(10, 3))
    plt.bar(np.arange(len(h_teinte)), h_teinte, width=1.0, color="black")
    for b in bornes[1:-1]:
        plt.axvline((b + decalage) % len(h_teinte), color="red", linestyle="--")
    plt.axvline(decalage, color="red", linestyle=":")  # endroit où on a coupé le cercle
    plt.xlabel("Teinte (degrés)")
    plt.ylabel("Nombre de pixels")
    plt.title(titre)
    plt.tight_layout()


def afficher_palette_hierarchique(niveaux, titre=""):
    # Une ligne par étape (teinte, saturation, intensité), comme la figure 4(e) de l'article
    fig, axes = plt.subplots(3, 1, figsize=(10, 2.4))
    for ax, nom in zip(axes, ["teinte", "saturation", "intensite"]):
        couleurs = np.array(niveaux[nom]) / 255
        ax.imshow(couleurs[None, :, :], aspect="auto")
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_ylabel(f"{nom}\n({len(couleurs)})", rotation=0, ha="right", va="center", fontsize=8)
    fig.suptitle(titre)
    fig.tight_layout()


def petits_tests():
    # Vérifications rapides des briques de base
    assert np.allclose(pav_decroissant([5, 3, 4, 2]), [5, 3.5, 3.5, 2])
    assert np.allclose(pav_croissant([5, 3, 4, 2]), [3.5, 3.5, 3.5, 3.5])
    H, S, I = rgb_vers_hsi(np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255], [128, 128, 128]]]))
    assert np.allclose(H[0, :3], [0, 120, 240]) and S[0, 3] == 0
    # Une bosse : unimodal. Deux bosses bien séparées : pas unimodal.
    x = np.arange(100)
    une_bosse = 1000 * np.exp(-(x - 50) ** 2 / 50)
    deux_bosses = une_bosse + 1000 * np.exp(-(x - 15) ** 2 / 20)
    assert est_unimodal_naif(une_bosse) and not est_unimodal_naif(deux_bosses)
    assert len(ftc(deux_bosses, est_unimodal_naif)) - 1 == 2
    print("Petits tests : OK")


if __name__ == "__main__":
    petits_tests()
    image = np.array(Image.open(IMAGE).convert("RGB"))
    graines, effectifs, niveaux, infos = palette_acopa(image, est_unimodal_naif)
    afficher_segmentation_teinte(infos["h_teinte"], infos["bornes_H"], infos["decalage"],
                                 f"Histogramme de teinte et segmentation FTC (seuil = {SEUIL})")
    afficher_palette_hierarchique(niveaux, f"Palette hiérarchique, version naïve (seuil = {SEUIL})")
    plt.show()
