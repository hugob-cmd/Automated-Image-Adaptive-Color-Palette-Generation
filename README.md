# Palette automatique de couleurs (projet 4IM01, Télécom Paris)

Implémentation et analyse de l'article : J. Delon, A. Desolneux, J.-L. Lisani, A.-B. Petro,
*Automatic Color Palette*, Inverse Problems and Imaging, 2007 (méthode **ACoPa**).

L'idée de la méthode : au lieu de fixer à la main le nombre de couleurs $K$, on segmente les
histogrammes de teinte, puis de saturation, puis d'intensité, en morceaux « unimodaux » au sens
d'un test statistique. Chaque morceau donne une couleur. Ces couleurs servent ensuite de centres
de départ à K-means.

Le rapport est le notebook **`rapport.ipynb`**, avec sa version PDF **`rapport.pdf`**.

## Organisation du dépôt

| Fichier | Contenu |
|---|---|
| `rapport.ipynb` / `rapport.pdf` | Rapport : méthodes, expériences, analyse critique, usage des agents conversationnels |
| `apprentissage/k-means.py`, `apprentissage/k-means-iterated.py` | Premiers essais de K-means « à la main » (double boucle sur les pixels), gardés comme traces de notre démarche |
| `median_cut.py` | Median cut (Heckbert) |
| `popularite.py` | Algorithme de popularité |
| `kmeans_rapide.py` | K-means vectorisé (utilisé partout ailleurs) |
| `comparaison.py` | Comparaison MSE / PSNR des méthodes classiques pour plusieurs $K$ |
| `acopa_simple.py` | Première version heuristique d'ACoPa (lissage + creux), sans le test de l'article |
| `segmentation_naive.py` | ACoPa avec un test d'unimodalité **à seuil réglé à la main** (HSI, PAV, FTC, hiérarchie teinte → saturation → intensité) |
| `segmentation_auto.py` | ACoPa avec le **test statistique a contrario** de l'article, puis K-means en RGB ou en Lab |
| `demo_resultats.py` | Images finales et palettes des méthodes automatiques sur 5 photos (enregistrées dans `resultats/`) |
| `images/` | Images de test (`astronaut`, `coffee`, `chelsea` viennent de `scikit-image`) |
| `resultats/` | Images produites par les scripts (non versionnées) |

Tous les scripts se lancent **depuis la racine du dépôt**, par exemple `python segmentation_auto.py`.
Dépendances : `pip install -r requirements.txt` (numpy, pillow, matplotlib, scikit-image, jupyter).

## Qui a codé quoi

| Fichier | Écrit par |
|---|---|
| `apprentissage/*.py`, `kmeans_rapide.py`, `popularite.py`, `comparaison.py`, `acopa_simple.py` | Hugo |
| `median_cut.py` | Aziz, avec l'aide de Claude (agent conversationnel) |
| `segmentation_naive.py`, `segmentation_auto.py`, `demo_resultats.py`, code du notebook | Claude (agent conversationnel), à partir de nos questions sur l'article ; relu et testé par Aziz |
| Corrections du 21/11 (imports, chemins, réorganisation) | Claude, à la demande d'Aziz |

Le détail de l'utilisation des agents conversationnels est dans la dernière partie du rapport.
