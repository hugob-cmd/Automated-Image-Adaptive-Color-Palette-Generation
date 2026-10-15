"""ACoPa (Delon et al., 2007), étapes 1 et 2 : espace HSI et Pool Adjacent Violators.

Contenu :
    rgb_to_hsi        : conversion RGB -> (teinte, saturation, intensité)
    chromatic_mask    : pixels "chromatiques" (hors cylindre des gris)
    hue_histogram     : histogramme de teinte à Q classes
    pav_decreasing    : meilleure approximation décroissante d'un histogramme
    pav_increasing    : meilleure approximation croissante (par symétrie)
    deviation_*       : écart relatif entre un histogramme et son approximation PAV
    is_unimodal       : test d'unimodalité à seuil fixe EPS (critère simple)

Usage :
    1. Renseigner IMAGE_PATH ci-dessous, puis : python acopa_etape1.py
       -> affiche l'image et son histogramme de teinte (et l'enregistre en PNG)
    2. Ou en ligne de commande : python acopa_etape1.py chemin/image.jpg
    3. Sans chemin (IMAGE_PATH = None) : tests + démo sur image synthétique
"""
import sys
import numpy as np

# ---------------------------------------------------------------------------
# À MODIFIER : chemin de l'image à analyser (None = tests sur image synthétique)
# Exemple : IMAGE_PATH = r"C:\Users\aziz\Images\fleur.jpg"
# ---------------------------------------------------------------------------
IMAGE_PATH = r"C:\Users\azizd\Downloads\poivron.jpg"  # ou None pour tests/démo
Q = 360  # nombre de classes de l'histogramme de teinte (1 classe = 1 degré)
EPS = 0.05  # seuil du test d'unimodalité : part de la masse déplacée par l'ajustement


# ---------------------------------------------------------------------------
# 1. Espace HSI
# ---------------------------------------------------------------------------
def rgb_to_hsi(image: np.ndarray):
    """Convertit une image RGB (h, w, 3) en trois tableaux (h, w) : H, S, I.

    I = (R + G + B) / 3
    S = distance du point (R, G, B) à l'axe des gris (norme de (R-I, G-I, B-I))
    H = angle (en degrés, dans [0, 360[) autour de l'axe des gris

    Remarque : l'article donne H = arccos(...), qui ne couvre que [0, pi].
    On utilise ici arctan2, qui couvre le cercle entier (rouge = 0 degré).
    Le choix de l'origine et du sens de l'angle n'a pas d'effet sur la
    segmentation, puisque l'histogramme de teinte est circulaire.
    """
    rgb = np.asarray(image, dtype=np.float64)
    R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    I = (R + G + B) / 3.0
    S = np.sqrt((R - I) ** 2 + (G - I) ** 2 + (B - I) ** 2)
    H = np.degrees(np.arctan2(np.sqrt(3.0) * (G - B), 2.0 * R - G - B)) % 360.0
    return H, S, I


def chromatic_mask(S: np.ndarray, Q: int = 360) -> np.ndarray:
    """Vrai pour les pixels chromatiques, faux pour ceux du cylindre des gris.

    Argument de l'article : à distance S de l'axe des gris, le cercle contient
    au plus 2*pi*S couleurs distinctes. Pour quantifier la teinte sur Q valeurs,
    il faut donc S > Q / (2*pi). En dessous, la teinte n'a pas de sens.
    """
    return S > Q / (2.0 * np.pi)


# ---------------------------------------------------------------------------
# 2. Histogramme de teinte
# ---------------------------------------------------------------------------
def hue_histogram(H: np.ndarray, S: np.ndarray, Q: int = 360) -> np.ndarray:
    """Histogramme de teinte à Q classes, sur les pixels chromatiques seulement.

    Les pixels gris sont exclus ici. L'algorithme complet de l'article les
    rattachera ensuite à un intervalle de teinte (étape 2 de ACoPa).
    """
    mask = chromatic_mask(S, Q)
    h, _ = np.histogram(H[mask], bins=Q, range=(0.0, 360.0))
    return h


# ---------------------------------------------------------------------------
# 3. Pool Adjacent Violators (estimateur de Grenander discret)
# ---------------------------------------------------------------------------
def pav_decreasing(h: np.ndarray) -> np.ndarray:
    """Meilleure approximation décroissante de h (moindres carrés).

    Principe : on parcourt h de gauche à droite en gardant une pile de blocs
    (valeur moyenne, taille). Dès que le dernier bloc est plus grand que
    l'avant-dernier (violation de la décroissance), on les fusionne en un bloc
    dont la valeur est la moyenne pondérée par les tailles. On recommence tant
    qu'il y a une violation. Complexité O(L).

    Exemple : pav_decreasing([5, 3, 4, 2]) -> [5, 3.5, 3.5, 2]
    """
    values, sizes = [], []
    for x in np.asarray(h, dtype=np.float64):
        values.append(x)
        sizes.append(1)
        while len(values) > 1 and values[-2] < values[-1]:
            s = sizes[-2] + sizes[-1]
            v = (values[-2] * sizes[-2] + values[-1] * sizes[-1]) / s
            values.pop(), sizes.pop()
            values[-1], sizes[-1] = v, s
    return np.repeat(values, sizes)


def pav_increasing(h: np.ndarray) -> np.ndarray:
    """Meilleure approximation croissante : on retourne h, on applique la
    version décroissante, on retourne le résultat."""
    return pav_decreasing(np.asarray(h)[::-1])[::-1]


def _pav_article(h: np.ndarray) -> np.ndarray:
    """Version "naïve" de l'annexe B : on remplace chaque partie croissante par
    sa moyenne, et on répète jusqu'à obtenir une suite décroissante.
    Sert uniquement à vérifier pav_decreasing."""
    r = np.asarray(h, dtype=np.float64).copy()
    while np.any(r[:-1] < r[1:]):
        i = 0
        while i < len(r):
            j = i
            while j + 1 < len(r) and r[j] <= r[j + 1]:
                j += 1
            r[i:j + 1] = r[i:j + 1].mean()
            i = j + 1
    return r


# ---------------------------------------------------------------------------
# 4. Test d'unimodalité (critère simple, seuil fixé à la main)
# ---------------------------------------------------------------------------
def _deviation(h: np.ndarray, fit: np.ndarray) -> float:
    """Écart relatif entre h et son ajustement : sum|h - fit| / (2 * N).

    Comme l'ajustement PAV conserve la masse totale N, cette quantité est la
    part des pixels que l'ajustement "déplace" pour rendre h monotone. Elle est
    comprise entre 0 (h est déjà monotone) et 1, et ne dépend pas de la taille
    de l'image.
    """
    h = np.asarray(h, dtype=np.float64)
    N = h.sum()
    if N <= 0:
        return 0.0
    return float(np.abs(h - fit).sum() / (2.0 * N))


def deviation_decreasing(h: np.ndarray) -> float:
    """Écart de h à sa meilleure approximation décroissante."""
    return _deviation(h, pav_decreasing(h))


def deviation_increasing(h: np.ndarray) -> float:
    """Écart de h à sa meilleure approximation croissante."""
    return _deviation(h, pav_increasing(h))


def deviation_unimodal(h: np.ndarray) -> float:
    """Écart de h à sa meilleure approximation unimodale (croissante puis
    décroissante).

    On essaie tous les points de coupure c : h[:c] est ajusté par une suite
    croissante, h[c:] par une suite décroissante. Concaténées, elles forment
    toujours une suite unimodale, et toute suite unimodale s'obtient ainsi.
    On garde la coupure qui donne le plus petit écart. Complexité O(L^2), soit
    environ 0,1 s pour L = 360.

    Exemple : [1, 3, 7, 10, 6, 3, 1] est déjà unimodal -> 0.
    """
    h = np.asarray(h, dtype=np.float64)
    N = h.sum()
    if len(h) < 2 or N <= 0:
        return 0.0
    best = np.inf
    for c in range(1, len(h)):
        gauche, droite = h[:c], h[c:]
        d = (np.abs(gauche - pav_increasing(gauche)).sum()
             + np.abs(droite - pav_decreasing(droite)).sum())
        best = min(best, d)
    return float(best / (2.0 * N))


def follows_decreasing(h: np.ndarray, eps: float = EPS) -> bool:
    """Hypothèse décroissante (définition 1 de l'article), critère simple."""
    return deviation_decreasing(h) <= eps


def follows_increasing(h: np.ndarray, eps: float = EPS) -> bool:
    """Hypothèse croissante (définition 1 de l'article), critère simple."""
    return deviation_increasing(h) <= eps


def is_unimodal(h: np.ndarray, eps: float = EPS) -> bool:
    """Hypothèse unimodale (définition 2 de l'article), critère simple :
    h est jugé unimodal si l'ajustement unimodal déplace au plus une fraction
    eps des pixels."""
    return deviation_unimodal(h) <= eps


# ---------------------------------------------------------------------------
# Tests et démo
# ---------------------------------------------------------------------------
def _tests():
    # --- HSI sur des couleurs pures
    px = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255], [128, 128, 128]]])
    H, S, I = rgb_to_hsi(px)
    print("Teintes (rouge, vert, bleu, gris) :", H.round(1)[0])
    print("Saturations                         :", S.round(1)[0])
    print("Intensités                          :", I.round(1)[0])
    assert np.allclose(H[0, :3], [0, 120, 240])
    assert np.isclose(S[0, 3], 0.0)

    # --- Seuil du cylindre des gris
    print(f"Seuil de saturation pour Q=360 : {360 / (2 * np.pi):.1f}")

    # --- PAV : exemple de l'article et propriétés
    h = np.array([5, 3, 4, 2])
    print("PAV décroissant de", h, "->", pav_decreasing(h))
    assert np.allclose(pav_decreasing(h), [5, 3.5, 3.5, 2])
    print("PAV croissant   de", h, "->", pav_increasing(h))
    assert np.allclose(pav_increasing(h), [3.5, 3.5, 3.5, 3.5])

    rng = np.random.default_rng(0)
    for _ in range(200):
        h = rng.integers(0, 50, size=rng.integers(2, 40)).astype(float)
        p = pav_decreasing(h)
        assert np.all(p[:-1] >= p[1:] - 1e-12)          # décroissant
        assert np.isclose(p.sum(), h.sum())               # masse conservée
        assert np.allclose(p, _pav_article(h))            # = version article
        q = pav_increasing(h)
        assert np.all(q[:-1] <= q[1:] + 1e-12)
    print("Tests PAV : OK (décroissance, masse conservée, accord avec l'annexe B)")

    # --- Test d'unimodalité : cas écrits à la main
    print()
    cas = {
        "décroissant      [9 7 7 4 2 1]          ": [9, 7, 7, 4, 2, 1],
        "unimodal         [1 3 7 10 6 3 1]       ": [1, 3, 7, 10, 6, 3, 1],
        "deux pics nets   [2 8 14 8 1 0 1 9 15 7 2]": [2, 8, 14, 8, 1, 0, 1, 9, 15, 7, 2],
    }
    for nom, h in cas.items():
        d = deviation_unimodal(np.array(h, dtype=float))
        print(f"{nom} écart = {d:.3f} -> unimodal : {d <= EPS}")
    assert deviation_unimodal(np.array([9, 7, 7, 4, 2, 1.])) == 0.0
    assert deviation_unimodal(np.array([1, 3, 7, 10, 6, 3, 1.])) == 0.0
    assert not is_unimodal(np.array(cas["deux pics nets   [2 8 14 8 1 0 1 9 15 7 2]"], dtype=float))

    # --- Histogrammes simulés (loi multinomiale = bruit de comptage des pixels)
    rng = np.random.default_rng(0)
    x = np.arange(41)

    def gauss(mu, s):
        return np.exp(-0.5 * ((x - mu) / s) ** 2)

    p_uni = gauss(20, 5) / gauss(20, 5).sum()
    p_bi = gauss(10, 3) + gauss(30, 3)
    p_bi = p_bi / p_bi.sum()
    for N in (500, 5000):
        d_uni = [deviation_unimodal(rng.multinomial(N, p_uni).astype(float)) for _ in range(20)]
        d_bi = [deviation_unimodal(rng.multinomial(N, p_bi).astype(float)) for _ in range(20)]
        print(f"N = {N:5d} pixels : un pic -> écart max {max(d_uni):.3f} | "
              f"deux pics -> écart min {min(d_bi):.3f}")
        assert min(d_bi) > EPS
        if N >= 5000:
            assert max(d_uni) <= EPS
    print("Tests d'unimodalité : OK")


def _synthetic_image(seed=0) -> np.ndarray:
    """Trois bandes de couleur (rouge, vert, bleu) + bruit + bande grise."""
    rng = np.random.default_rng(seed)
    img = np.zeros((120, 120, 3))
    img[:100, :40] = (200, 40, 40)
    img[:100, 40:80] = (40, 180, 60)
    img[:100, 80:] = (50, 60, 200)
    img[100:] = (128, 128, 128)
    img += rng.normal(0, 10, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def _demo(img: np.ndarray, Q: int = 360):
    H, S, I = rgb_to_hsi(img)
    mask = chromatic_mask(S, Q)
    h = hue_histogram(H, S, Q)
    print(f"Pixels chromatiques : {mask.sum()} / {mask.size} "
          f"({100 * mask.mean():.0f} %), gris : {(~mask).sum()}")
    print("Pixels par secteur de 60 degrés :", h.reshape(6, -1).sum(axis=1))
    # Lissage "décroissant" sur le flanc droit du pic principal
    k = int(h.argmax())
    flanc = h[k:k + 30]
    print(f"Pic principal à {k} degrés ; flanc droit (30 classes) :")
    print("  brut  :", flanc[:10])
    print("  PAV   :", pav_decreasing(flanc)[:10].round(1))


def _demo_unimodal(h: np.ndarray, eps: float = EPS):
    """Applique le test d'unimodalité à une fenêtre autour du pic principal,
    puis à l'histogramme entier."""
    k = int(h.argmax())
    lo, hi = max(0, k - 20), min(len(h), k + 21)
    d_fen = deviation_unimodal(h[lo:hi])
    d_tot = deviation_unimodal(h)
    print(f"Fenêtre [{lo}, {hi}[ autour du pic principal : écart = {d_fen:.3f} "
          f"-> unimodal : {d_fen <= eps}")
    print(f"Histogramme entier ({len(h)} classes)         : écart = {d_tot:.3f} "
          f"-> unimodal : {d_tot <= eps}")


def plot_hue_histogram(img: np.ndarray, Q: int = 360, out_png: str = "histogramme_teinte.png"):
    """Affiche l'image (à gauche) et son histogramme de teinte (à droite).

    Chaque barre est colorée avec la teinte qu'elle représente (saturation et
    luminosité maximales), pour lire directement quelle couleur correspond
    à quel pic. Rappel : l'axe est circulaire (0 degré = 360 degrés = rouge).
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import hsv_to_rgb

    H, S, I = rgb_to_hsi(img)
    mask = chromatic_mask(S, Q)
    h = hue_histogram(H, S, Q)

    centres = (np.arange(Q) + 0.5) * 360.0 / Q
    couleurs = hsv_to_rgb(np.stack([centres / 360.0, np.ones(Q), np.ones(Q)], axis=1))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5),
                                   gridspec_kw={"width_ratios": [1, 2]})
    ax1.imshow(img)
    ax1.set_title(f"Image ({img.shape[1]} x {img.shape[0]})")
    ax1.axis("off")

    ax2.bar(centres, h, width=360.0 / Q, color=couleurs, edgecolor="none")
    ax2.set_xlim(0, 360)
    ax2.set_xticks(range(0, 361, 60))
    ax2.set_xlabel("Teinte (degrés)")
    ax2.set_ylabel("Nombre de pixels")
    ax2.set_title(f"Histogramme de teinte, Q = {Q} classes\n"
                  f"{100 * mask.mean():.0f} % de pixels chromatiques "
                  f"(seuil S > {Q / (2 * np.pi):.1f}), "
                  f"{100 * (1 - mask.mean()):.0f} % exclus (gris)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    print(f"Figure enregistrée : {out_png}")
    plt.show()


def _load_image(path: str) -> np.ndarray:
    from PIL import Image
    return np.array(Image.open(path).convert("RGB"))


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) == 2 else IMAGE_PATH
    if path:
        img = _load_image(path)
        _demo(img, Q)
        H, S, _ = rgb_to_hsi(img)
        _demo_unimodal(hue_histogram(H, S, Q), EPS)
        plot_hue_histogram(img, Q)
    else:
        _tests()
        print()
        _demo(_synthetic_image())