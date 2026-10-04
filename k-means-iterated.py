import numpy as np
from PIL import Image

#image_finale = Image.fromarray(mon_tableau)
#Fonction pour rendre une image à partir d'un tableau 
#image_finale.show() pour montrer l'image 
#Fonction pour prendre une imaghe 
# image = Image.open("mon_image.jpg")

# S'assurer que l'image est bien en mode RGB (et non RGBA ou Noir et Blanc)
#image = image.convert("RGB")

# 2. Convertir l'image en un tableau mathématique (matrice NumPy)
# Ce tableau aura la forme (Hauteur, Largeur, 3)
#pixels = np.array(image)

# 3. Lire le triplet RGB du pixel situé tout en haut à gauche (Ligne 0, Colonne 0)
#premier_pixel = pixels[0, 0]
#print(f"Le triplet RGB du premier pixel est : {premier_pixel}")
# Résultat attendu : par exemple [255 128 0]

image= Image.open("Poivrons.jpeg")
image.show()
pixels=np.array(image)

print(pixels[0,0])
'''centres_base = [
    [0, 0, 0],
    [255, 255, 255],
    [255, 0, 0],
    [0, 255, 0],
    [0, 0, 255],
    [255, 255, 0],
    [0, 255, 255],
    [255, 0, 255],
    [128, 128, 128],
    [255, 128, 0],
    [128, 0, 128],
    [0, 128, 128]
    ]
    '''
#En prenant des pixels qui n'ont rien à voir ave l'image on se retroouve avec une image completement marron
'''centres_base = [
    pixels[10,10],
    pixels[100,10],
    pixels[10,100],
    pixels[200,100],
    pixels[200,100],
    pixels[200,200],
    pixels[100,100],
    pixels[60,60],
    pixels[128,128],
    pixels[500,400],
    pixels[300,300],
    pixels[250,250],
    ]
'''
centres_base = [
    # Ligne du haut (Y = 85)
    pixels[85, 64],
    pixels[85, 192],
    pixels[85, 320],
    pixels[85, 448],
    pixels[256, 64],
    pixels[256, 192],
    pixels[256, 320],
    pixels[256, 448],
    pixels[426, 64],
    pixels[426, 192],
    pixels[426, 320],
    pixels[426, 448]
]

new_centres=[[]for i in range (0,len(centres_base))]
print(new_centres)

dims = pixels.shape
labels = np.zeros((dims[0], dims[1]), dtype=int)
new_pixels = np.zeros_like(pixels)


for i in range (0,dims[0]):
    for j in range (0,dims[1]):
        d=np.linalg.norm(pixels[[i,j]] - centres_base[0])
        for k in range(len(centres_base)) :
            distance = np.linalg.norm(pixels[[i,j]] - centres_base[k])
            if distance<=d:
                d=distance
                new_centres[k].append(pixels[i,j])
                labels[i,j]=k


for k in range(len(new_centres)):
    if new_centres[k]!=[]:
        a=np.mean(new_centres[k],axis=0)
        print(a)
        new_centres[k]=a
    else:
        print(f"Groupe {k} n'a aucun pixel associé")

for i in range (0,dims[0]):
    for j in range (0,dims[1]):
        groupe=labels[i,j]
        new_pixels[i,j]=new_centres[groupe]

new_pixels = new_pixels.astype(np.uint8)

image_finale = Image.fromarray(new_pixels)
image_finale.show()

