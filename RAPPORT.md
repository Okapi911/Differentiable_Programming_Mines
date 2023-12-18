# <center>Image Captioning</center>

## I. Introduction

Pour notre sujet, nous avions envie de mélanger plusieurs types de modèles. Si l'idée était au départ de partir d'une descritpion pour générer une image, nous avons choisi d'être réaliste et de nous tourner vers de la description d'images.
Au dela de son utilité pour les personnes mal-voyantes, la description d'image pourra trouver d'autres utilités, notamment dans le milieu du tourisme où l'on peut imaginer l'utilisation d'ia pour remplacer les plaques aux points d'intérets, souvent disponible seulement dans une ou deux langues.

## II. Choix du modèle

## III. Choix du dataset

Intiallement, le choix de notre dataset s'était porté sur un dataset disponible sur Huggingface : https://huggingface.co/datasets/yuvalkirstain/pexel_images_lots_with_generated_captions. Ce jeu de données comporte 7999 photos normalisées qui font toutes 512x512 accompagnées de leur descritpion, une simplicité de téléchargement grâce au package datasets et les images semblaient toutes de bonne qualité.
Cependant, certaines des descriptions avaient été generées automatiquement (descritpions n'ayant pas de sens ou avec des répétitions de mots) et les images étant principalement des portraits nous n'étions pas sûrs d'utiliser ce dataset.

Le dataset proposé avec le tutoriel s'est averé être beaucoup plus diversifé, et proposer un nombre d'images bien plus importants dont les descriptions semblaient avoir été annotées manuellement. Sachant que notre code était déjà rédigé pour ce dataset, nous avons décidé de le garder. Nous avons limité sa taille (32 000) et l'avons séparé en test (70%), validation (12%) et test (18%) pour garder toujours un nombre d'images divisible par 32.

## IV. Entrainement et validation

## V. Test

Pour analyser le test nous allons nous intéresser à des exemples testés sur le modèle entrainé avec 40 epochs.
Je vais d'abord présenter des images choisies en dehors du dataset pour que l'on voit la performance de notre modèle dessus

<center>
    <figure>
        <img src="Examples/boat.png" alt="Boat">
        <figcaption>a man is standing on a rock overlooking a lake.</figcaption>
    </figure>
    <figure>
        <img src="Examples/bus.png" alt="Bus">
        <figcaption>a man in a red shirt and black pants is sitting on a bench.</figcaption>
    </figure>
    <figure>
        <img src="Examples/child.jpg" alt="Child">
        <figcaption>a little girl in a pink shirt is walking on a sidewalk.</figcaption>
    </figure>
    <figure>
        <img src="Examples/dog.jpg" alt="Dog">
        <figcaption>a dog is running through the water.</figcaption>
    </figure>
    <figure>
        <img src="Examples/horse.png" alt="Horse">
        <figcaption>a man is standing on a rock overlooking a lake.</figcaption>
    </figure>
</center>

On se rend très rapidement compte que le modèle n'est pas performant sur ces images. Le modèle est incapable d'identifier un bateau ou un bus, on obtient deux fois la même description pour des images très différentes et certaines tendance s'identifient presque dans les descriptions proposées.

Si l'on s'intéresse aux images présentent dans la base d'entrainement, on peut déjà identifier quelques causes :

<center>
    <figure>
        <img src="Examples/ex_entrainement.png" alt="8 images de la base de train">
        <figcaption>Images tirées aléatoirement de la base de test</figcaption>
    </figure>
</center>

En effet, sur ces photos, on se rend rapidement compte qu'il y a beaucoup d'humains : cela explique les faiblesses du modèle pour identifier le bateau ou le bus. Si l'on s'intéresse aux descriptions d'entrainement, on identifie également des tendances dans l'utilisation des mots. Les mots tels que "a" "in" et "is" aparaissent très fréquemment dans les descriptions mais aussi dans la langue anglaise, il est donc normal qu'ils soient beaucoup utilisés. Cependant, les mots tels que "man", "rock" ou encore "shirt" apparraissent beaucoup dans le dataset, si l'on regarde leur colocation on voit beaucoup apparaitre "a man" suivi par "in" qui sera souvent suivi par "a red shirt". De même la colocation "rock formation" est assez fréquente, même si elle n'apparait pas dans ces exemples.
En fait, on se rend compte que quand le modèle n'arrive pas à savoir ce qu'il voit précisément, il a recours à des "phrases bateau".

On va maintenant s'intéresser à des exemples issus de la base de test pour voir si les performances du modèle sont meilleures.

## VI. Ajustement des hyperparamètres

## VII. Préconisation
