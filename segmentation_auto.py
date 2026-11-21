# --------------------------------------------------------------------------
# Projet 4IM01 - Palette automatique de couleurs (article ACoPa, Delon et al. 2007)
#
# Qui a écrit ce fichier :
#   Code écrit par Claude (agent conversationnel d'Anthropic) à partir de l'article
#   et de la référence [10] (Delon, Desolneux, Lisani, Petro, "A nonparametric approach
#   for histogram segmentation", IEEE TIP 2007), puis relu et testé par Aziz.
#
# VERSION AUTOMATIQUE de la segmentation :
#   on garde tout ce qui est dans segmentation_naive.py (HSI, PAV, FTC, hiérarchie
#   teinte -> saturation -> intensité) mais on remplace le seuil réglé à la main
#   par le test statistique "a contrario" de l'article. Le seul paramètre est EPSILON,
#   le nombre moyen de fausses détections qu'on accepte (EPSILON = 1 dans [10]).
#   Ensuite les couleurs trouvées servent de centres de départ à K-means (annexe A),
#   dans l'espace RGB ou dans l'espace CIE Lab.
#
# Lancer depuis la racine du dépôt : python segmentation_auto.py
# --------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.color import rgb2lab

from segmentation_naive import (pav_croissant, pav_decroissant, pics_candidats, ftc,
                                palette_acopa, afficher_segmentation_teinte,
                                afficher_palette_hierarchique)
from kmeans_rapide import kmeans, image_depuis_labels

IMAGE = "images/Poivrons.jpeg"
EPSILON = 1.0   # nombre de fausses alarmes accepté (valeur de [10])


# ==========================================================================
# 1. Test a contrario : "h suit-il la loi p ?"
# ==========================================================================
def entropie_relative(r, p):
    # H(r, p) = r log(r/p) + (1-r) log((1-r)/(1-p)), avec la convention 0 log 0 = 0
    # C'est la "distance" entre une proportion observée r et une proportion attendue p.
    petit = 1e-12
    p = np.clip(p, petit, 1 - petit)
    terme1 = np.where(r > 0, r * np.log(np.maximum(r, petit) / p), 0)
    terme2 = np.where(r < 1, (1 - r) * np.log(np.maximum(1 - r, petit) / (1 - p)), 0)
    return terme1 + terme2


def suit_la_loi(h, loi, L_total):
    # h : histogramme observé (N pixels), loi : histogramme attendu (même masse N).
    # Pour chaque intervalle [a, b], on compare la proportion observée r(a,b) à la
    # proportion attendue p(a,b). Sous l'hypothèse "h suit la loi", le nombre de pixels
    # dans [a, b] suit une loi binomiale B(N, p(a,b)). La probabilité d'observer un écart
    # aussi grand est majorée par exp(-N H(r, p)) (borne de Hoeffding).
    # On teste tous les intervalles, donc on multiplie par leur nombre L(L+1)/2 :
    #     NFA([a,b]) = L(L+1)/2 * exp(-N H(r(a,b), p(a,b)))
    # Si NFA < EPSILON pour au moins un intervalle, on rejette l'hypothèse.
    # Autrement dit on rejette si   H(r, p) > log(L(L+1) / (2 EPSILON)) / N.
    N = h.sum()
    if N == 0:
        return True
    # Sommes cumulées : nombre de pixels dans [a, b] = cumul[b+1] - cumul[a]
    cumul_h = np.concatenate([[0], np.cumsum(h)]) / N
    cumul_loi = np.concatenate([[0], np.cumsum(loi)]) / N
    r = cumul_h[None, 1:] - cumul_h[:-1, None]      # r[a, b] (on ne garde que b >= a)
    p = cumul_loi[None, 1:] - cumul_loi[:-1, None]
    triangle = np.triu(np.ones((len(h), len(h)), dtype=bool))
    H = entropie_relative(np.clip(r[triangle], 0, 1), p[triangle])
    seuil = np.log(L_total * (L_total + 1) / (2 * EPSILON)) / N
    return not np.any(H > seuil)


# ==========================================================================
# 2. Hypothèses croissante, décroissante, unimodale (définitions 1 et 2 de l'article)
# ==========================================================================
def suit_hypothese_croissante(h, L_total):
    # h suit l'hypothèse croissante si le test accepte la loi donnée par son
    # estimateur de Grenander croissant (calculé par PAV)
    return suit_la_loi(h, pav_croissant(h), L_total)


def suit_hypothese_decroissante(h, L_total):
    return suit_la_loi(h, pav_decroissant(h), L_total)


def fabrique_test_unimodal(L_total):
    # On fabrique la fonction de test pour un histogramme de L_total classes
    # (L_total sert dans le nombre d'intervalles testés)
    def est_unimodal_auto(h):
        h = np.asarray(h, dtype=np.float64)
        if len(h) < 3 or h.sum() == 0:
            return True
        # Unimodal sur [a, b] s'il existe un pic c tel que h soit croissant sur [a, c]
        # et décroissant sur [c, b] (au sens du test). On essaie les pics les plus hauts.
        for c in pics_candidats(h):
            if suit_hypothese_croissante(h[:c + 1], L_total) and suit_hypothese_decroissante(h[c:], L_total):
                return True
        return False
    return est_unimodal_auto


# palette_acopa appelle est_unimodal(h) sans donner la taille de l'histogramme complet.
# On prend donc L_total = 360 (le plus grand histogramme, celui de la teinte) pour tous
# les niveaux : c'est un peu plus sévère pour la saturation (210 classes) et l'intensité
# (256 classes), mais ça reste simple.
est_unimodal_auto = fabrique_test_unimodal(360)


# ==========================================================================
# 3. K-means initialisé avec les graines d'ACoPa (annexe A), en RGB ou en Lab
# ==========================================================================
def acopa_kmeans(image, graines, espace="RGB"):
    # Renvoie l'image quantifiée et la palette finale (couleurs RGB moyennes des groupes)
    h, w, _ = image.shape
    pixels = image.reshape(-1, 3).astype(np.float64)
    if espace == "Lab":
        # En Lab, les distances euclidiennes sont proches des différences perçues par l'oeil
        pixels_travail = rgb2lab(image / 255.0).reshape(-1, 3)
        graines_travail = rgb2lab(graines[None, :, :] / 255.0)[0]
    else:
        pixels_travail = pixels
        graines_travail = graines
    # Seuil d'arrêt 1 comme dans l'annexe A
    _, labels = kmeans(pixels_travail, graines_travail, seuil=1.0, afficher=False)
    # Palette finale = couleur RGB moyenne de chaque groupe (comme dans l'article)
    palette = []
    for k in range(len(graines)):
        if np.any(labels == k):
            palette.append(pixels[labels == k].mean(axis=0))
        else:
            palette.append(graines[k])  # groupe vide : on garde la graine
    palette = np.array(palette)
    return image_depuis_labels(palette, labels).reshape(h, w, 3), palette


def acopa_complet(image, espace="Lab", afficher=True):
    # Tout l'algorithme : palette hiérarchique automatique puis K-means
    graines, effectifs, niveaux, infos = palette_acopa(image, est_unimodal_auto, afficher)
    resultat, palette = acopa_kmeans(image, graines, espace)
    return resultat, palette, graines, niveaux, infos


def petits_tests():
    x = np.arange(100)
    une_bosse = np.round(1000 * np.exp(-(x - 50) ** 2 / 50))
    deux_bosses = une_bosse + np.round(1000 * np.exp(-(x - 15) ** 2 / 20))
    assert est_unimodal_auto(une_bosse) and not est_unimodal_auto(deux_bosses)
    # Une très petite bosse (30 pixels) à côté d'une grosse (environ 12 000 pixels)
    # doit être gardée : c'est le cas de la coccinelle dans l'article
    petite = une_bosse + np.round(30 * np.exp(-(x - 85) ** 2 / 2))
    assert len(ftc(petite, est_unimodal_auto)) - 1 == 2
    print("Petits tests : OK")


if __name__ == "__main__":
    petits_tests()
    image = np.array(Image.open(IMAGE).convert("RGB"))
    resultat, palette, graines, niveaux, infos = acopa_complet(image, "Lab")
    print(f"K = {len(palette)} couleurs trouvées automatiquement")
    afficher_segmentation_teinte(infos["h_teinte"], infos["bornes_H"], infos["decalage"],
                                 "Histogramme de teinte et segmentation FTC (test a contrario)")
    afficher_palette_hierarchique(niveaux, "Palette hiérarchique (version automatique)")
    Image.fromarray(resultat).save("resultats/acopa_lab.png")
    plt.figure()
    plt.imshow(resultat)
    plt.axis("off")
    plt.title(f"ACoPa + K-means (Lab), K = {len(palette)}")
    plt.show()
