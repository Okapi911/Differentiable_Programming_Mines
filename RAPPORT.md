# <center>Image Captioning</center>

## I. Introduction

- Intéret personnel CNN + NLP. LSTM pas préentrainé 100% sur notre modèle
- Technologie utile puisque peut être utilisé pour les mal voyants. Possibilité d'imaginer que ça puisse également s'utiliser pour de la description d'objets d'autres pays ou d'autres cultures.

## II. Choix du modèle

## III. Choix du dataset

- Intuition : Dataset huggingface + image de bonne qualité, toutes au même format, accessible facilement - 7999 images, certaines captions semblent ecritent par ordinateur, manque de diversité (beaucoup de portraits d'humains)
- Choix final : Dataset flicker8k utilisé dans la vidéo + quantité d'images plus importante (37 000), plus de diversité, captions semblent rédigée par des humains
- En plus de ça, manque de compatibilité de la base de données huggingface avec le code trouvé, choix de ne pas perdre de temps à essayer d'adapter le code pour se concentrer sur l'analyse.

## IV. Entrainement et validation

## V. Test

## VI. Ajustement des hyperparamètres

## VII. Préconisation
