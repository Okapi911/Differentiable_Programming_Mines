# Generation de description d'image.

## Choix de la base de données :

Database Hugging face contenant 7999 éléments, majoritairement des portraits d'humains, mais également quelques portraits d'animaux. La base ne fournit que du train, nous ferons donc initialement de la cross-validation, puis nous chercherons peut-être par la suite une base de test.

Exemple d'image de la base de données :
![image.jpg](image.jpg)

Dont la description est : "portrait of a woman wearing a yellow beanie hat and sunglasses, holding a pair of white sunglasses".

## Choix du modèle :

<div align="justify">
Nous avons choisi d'utiliser une combinaison d'un réseau de convolution CNN qui prépare les embeddings à fournir à un modèle de NLP. En nous inspirant du projet github suivant : https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/more_advanced/image_captioning créé par l'utilisateur aladdinpersson, nous avons décidé de réaliser cette partie NLP par la mise en place d'un réseau de LSTM (Long Short Term Memory).

Ainsi notre projet consiste à la fois à fine-tune un CNN performant pour construire des embeddings d'images qui pourront être réutilisés par un modèle NLP, et à entraîner ce nouveau modèle constitué d'un LSTM puis d'une couche linéaire pour générer des descriptions d'images. Cette dernière couche linéaire a pour but de choisir le token (ou mot) le plus probable à partir de la séquence d'embeddings fournie par le LSTM.

<figure><center>
<img src="./Autres figures rapport/model.png">
<center><label>Schématisation du fonctionnement de notre modèle</label></center>
</center></figure>

Décrivons plus en détail le fonctionnement de notre modèle. Tout d'abord, les images sont transformées au moment de la conception des dataloaders pour avoir mêmes dimensions et propriétés indépendamment du format d'origine (en particulier on veut pouvoir traiter des png et des jpg sans distinction). Les images sont alors converties en tenseurs et normalisées. On forme ensuite les batchs d'images avec leurs descriptions associées en utilisant la fonction get_loader que nous avons implémentée. Elle a été spécifiquement conçue pour notre jeu de données qui s'organise en un dossier contenant les images et un fichier texte au format csv 2 colonnes contenant les descriptions et le path des images correspondantes. Dans le cadre de ce projet, nous avons pris l'initiative de répartir les données en base d'entraînement, de validation et de test avec respectivement 70%, 15% et 15% des données.

Nous avions initialement prévu de réaliser de la data augmentation notamment en faisant varier la rotation des images et la luminosité. Cependant, ces choix ne nous ont finalement pas parus pertinents car ils ne sont pas représentatifs des applications réelles potentielles du projet et au lieu de régulariser le modèle, ils ont eu tendance à le dégrader dans les premières phases de tests. La seule forme de data augmentation que nous avons finalement conservé est de rogner aléatoirement les images de format 356 par 356 pixels en un carré de 299 par 299 pixels. Nous garantissons ainsi que le modèle ne soit pas trop sensible à la position de l'objet dans l'image. Par exemple, si à notre insu les images ayant des chiens dans le dataset les représentent toujours dans la partie gauche de l'image, ce cropping aléatoire permettrait de recentrer cette position sur certaines images lorsque c'est la partie droite qui est rognée. On contribue ainsi à rendre notre modèle plus robuste avec des inputs plus petits sans perte de performance notable.

Pour pouvoir plus tard calculer une loss par CrossEntropy en sortie du LSTM, il faut dès la création des dataloaders représenter les descriptions sous forme de tenseurs contenant la séquence des tokens. Pour cela, nous avons conçu une classe Vocabulary qui permet de construire un vocabulaire à partir des descriptions du jeu de données. Tout token (symbolisant environ une syllabe) est représenté par un entier.

<figure><center>
<img src="./Autres figures rapport/inceptionv3.png">
<center><label>Représentation du modèle Inception V3</label></center>
</center></figure>

Une fois les dataloaders et le vocabulaire créés, le pre-processing est terminé : on dispose des batchs d'inputs à donner au modèle (avec leurs labels). Développons donc plus en détail le modèle à proprement dit, à commencer par le CNN pré-entraîné. Le modèle que nous allons fine-tune est InceptionV3, un réseau de convolution très performant qui a été pré-entraîné sur ImageNet. Ses résultats sur le corpus dépassent 78,1% de précision avec "seulement" 24 millions de paramètres. Il est donc plus performant que le modèle usuel resnet50 et beaucoup moins coûteux que les modèles ayant des performances à peine supérieures comme resnet152. La figure suivante permet de comparer divers modèle de classification d'image et leur précision sur le corpus ImageNet. Les meilleurs modèles sont dans le coin supérieur gauche (bonne précision, peu de paramètres).

<figure><center>
<img src="./Autres figures rapport/image_recognition_benchmark.png">
<center><label>Comparaison des performances de différents modèles de classification d'images</label></center>
</center></figure>

Pour l'adapter à notre tâche et faire du fine-tuning, on remplace la partie fully connected (MLP) permettant la classification en fin de modèle par une unique couche linéaire dont le but est de produire à partir du feature vector (synthèse vectorielle de l'ensemble des représentations finales des inputs dans le modèle au sortir de la dernière convolution) des embeddings de la taille souhaitée pour le LSTM (la taille du vocabulaire).

Ces embeddings sont ensuite passés dans un LSTM qui va générer une nouvelle séquence de même dimension que le vocabulaire. Enfin, une couche linéaire permettra de prédire le token le plus probable à partir de cette séquence d'embeddings.

Le modèle est entraîné par backpropagation et descente stochastique avec une loss par CrossEntropy (puisqu'on fait à chaque élément de la séquence / du descriptif une classification sur le bon token du langage). En plus de la SGD, la régularisation du modèle est assurée par un Dropout sur les couches linéaires et sur les embeddings en entrée du LSTM.

## Entraînement et Validation

Dans cette partie, nous allons vous présenter les résultats de notre entraînement le plus poussé ; c'est aussi celui qui nous a donné les meilleurs résultats.

Les hyperparamètres entrant en jeu dans le modèle sont le taux d'apprentissage, la probabilité de Dropout et la taille des batchs dans les différents Dataloader. Pour cet entraînement spécifique, les valeurs ont respectivement été fixées à 3e-4 ; 0.2 et 32 images par batch. Nous reviendrons dans une partie ultérieure sur l'importance des hyperparamètres et les tests que nous avons pu faire avec d'autres valeurs.

L'entraînement s'est fait sur 40 epochs sans early stopping, mais nous aurions initialement souhaité poursuivre l'entraînement sur 100 epochs comme avait pu le faire notre inspiration (lien du github dans la partie dédiée à la description du modèle). La raison pour laquelle nous avons limité la "durée" d'entraînement est uniquement liée aux faibles capacités des machines sur lesquelles nous avons travaillées. En effet ces dernières ne disposaient pas de GPU, et chaque epoch (pour apprentissage puis validation) prenait environ 10 minutes, ce qui nous limitait grandement puisque de plus ce n'était pas nos machines personnelles (nous devions donc interrompre l'apprentissage la nuit ou quand nous n'étions pas dans la salle du matériel informatique).

<figure><center>
<img src="./Autres figures rapport/Entrainement_40x700_batchs.png">
<center><label>Evolution de la loss à chaque batch succesif traité par le modèle pendant l'entraînement</label></center>
</center></figure>

La figure précédente représente l'évolution de la Loss par Batch sur le dataset d'entraînement tout au long de la phase d'apprentissage. L'axe des abscisses s'étend donc sur 40\*700 = 28 000 batchs. On remarque que la courbe commence à converger en fin d'entraînement, mais le minimum n'a pas encore été atteint. Avant même de faire apparaître la courbe de validation on peut intuiter que l'on n'est pas dans une situation d'hyperapprentissage puisque certains batchs de la base d'entraînement ont encore des mauvais scores en fin d'entraînement (pour lesquels la valeur de la loss est plus élevée, comparable à la moyenne 7000 batchs auparavant).

<figure><center>
<img src="./Autres figures rapport/entrainement_40_epochs.png">
<center><label>Courbes de loss sur les datasets d'entraînement et de validation du modèle sur 40 epochs</label></center>
</center></figure>

La figure ci-dessus représente les deux courbes de loss (par image en entrée) au long des 40 epochs d'entraînement sur le dataset d'entraînement (courbe bleue) et sur celui de validation (courbe orange). On constate que très vite la courbe de validation présente des valeurs supérieures à celle de training et converge vers une valeur de loss supérieure, ce qui correspond bien aux attentes générales d'un modèle d'IA. En effet puisque le modèle est entraîné sur la basse d'entraînement il finit par la mémoriser même sans l'apprendre par coeur et aura donc de meilleurs scores que pour de la généralisation sur des "nouvelles" données (celles de la base de alidation n'entrent jamais en mémoire puisque le modèle est réglé en mode évaluation avant chaque validation donc pas de backpropagation).

Comme on l'observait plus tôt sur la figure précédente, la loss sur la base d'entraînement n'a pas encore convergé en 40 epochs, et sur cette dernière courbe on voit encore mieux que le minimum est lointain puisqu'on a seulement un début d'aplatissement. Comme de plus la courbe de loss sur la base de validation ne remonte pas en fin d'entraînement cela signifie qu'on n'est pas dans un cas de sur-apprnetissage et on aurait pu se permettre de prolonger l'entraînement. Notons néanmoins que cette courbe semble avoir convergé sur la deuxième partie de l'entraînement (après 20 epochs). Ainsi le score de validation des tokens augmenterait peu même si l'on prolongeait l'entraînement, toutefois comme on est sur un modèle de langage il aurait pu être intéressant de voir si les descroiptions produites même incorrectes faisaient "plus de sens". Par cette dernière expression on entend que des synonymes auraient pu être utilisés pour un sens similaire, ou qu'en cas de véritable erreur on soit capable de comprendre le raisonnement erronné du modèle (plus sur ce point en partie suivante).

Pour conclure cette partie sur la période d'apprentissage, on pourra ajouter que lors de cette expérience nous étion passé en tuning complet du modèle convolutionnel InceptionV3 : après avoir effectué des tests comparatifs de modèles entraînés sur 10 epochs nous avons constatés que la capacité de généralisation du modèle s'améliorait si on autorisait la backpropagation jusqu'aux couches de convolution plutôt que de l'utiliser uniquement pour le fine tuning de la nouvelle partie fully-connected que nous avons ajouté (qui n'est pas une tête de classification puisqu'au contraire on retarde cette tâche en sortie du LSTM).

## Influence des Hyperparamètres

Comme indiqué dans la partie traitant de l'entraînement de notre modèle le plus poussé, nous avons un total de 3 hyperparamètres dans notre modèle (outre le nombre d'epochs d'apprentissage). Ils correspondent au taux d'apprentissage, à la probabilité de Dropout, et la taille des Batch dans les différents dataloaders (ici comme souvent cette taille sera la même pour entraînement, validation et test).

Le taux d'apprentissage et la probabilité de Dropout auront été choisis après avoir réalisé des études documentaires dans le domaine spécifique de l'image captioning et en ayant entraîné quelques modèles sur un nombre réduit d'epochs. Ces tests nous ont permis d'observer que les meilleures valeurs de Dropout sont dans l'intervalle [0.11 ; 0.24] pour avoir des loss par image de test inférieures en moyenne à 0.15.

Concernant le choix du taux d'apprentissage, on s'est demandé pourquoi prendre une valeur si basse quand le score de validation convergeait encore vers des valeurs que nous jugions trop hautes en fin d'entraînement (avec une qualité discutable des échantillons de tests comme montré précédemment) et puisque la training loss ne converge pas pendant nos entraînements.

Ainsi, même si nos recherches documentaires s'accordaient pour des valeurs du learning rate de l'ordre de 10^-4 nous avons réalisé un test en l'augmentant à 10^-3 pour un entraînement de 10 epochs. Sur cet entraînement nous avons également fait le choix d'augmenter la régularisation pour lisser les courbes en profitant de la SGD avec des batchs plus réduits (16 images par batch). Les résultats (comparés aux 10 premières époques du modèle précédemment décrits) ont été les suivants :

<figure><center>
<img src="./Autres figures rapport/Entraînements comparés.png">
<center><label>Courbes de loss sur les datasets d'entraînement sur l'ancien modèle (courbe orange : lr = 3e-4 ; batch_size=32) et sur le nouveau (en bleu ; lr = 1e-3 ; batch_size = 16).</label></center>
</center></figure>

<figure><center>
<img src="./Autres figures rapport/Validations_comparées.png">
<center><label>Courbes de loss sur les datasets de validation sur l'ancien modèle (en orange ; lr = 3e-4 ; batch_size=32) et sur le nouveau (en bleu ; lr = 1e-3 ; batch_size = 16).</label></center>
</center></figure>

Ces courbes montrent bien que le modèle précédent était meilleur : la loss à l'apprentissage y était inférieure à chaque epoch et l'on avait aucun surapprentissage. En revanche avec le nouveau modèle la loss sur le dataset d'entraînement semble se diriger vers un moins bon minimum (puisqu'on semble converger vers une valeur supérieure de loss). De plus ce nouveau modèle est en surapprentissage comme démontré par la courbe de loss à chaque validation : la loss diminue sur seulement 4 epochs avant d'augmenter de nouveau ce qui signifie que la capacité de prédiction diminue très vite. Pour empêcher ce surapprentissage on aurait pu augmenter encore davantage la régularisation notammentr au travers du dropout. En l'état les seules circonstances dans lesquelles ce deuxième modèle est meilleur est un entraînement de 4 epochs ou moins puisqu'alors on n'a pas de surapprentissage et on a des meilleurs scores de validation (car un learning rate plus élevé a permis de changer plus vite les paramètres du modèle). Néanmoins on conçoit bien que l'on n'a aucune raison de faire ce choix : en 4 epochs la qualité prédictive du modèle est tout au mieux médiocre (toutes les images de chiens de la partie précédente conduisant par exemple à la même description qui ne s'applique à aucune des : "a dog is running in the snow").

## Préconisations

Une question fondamentale subsiste après ces analyses : comment améliorer la capacité de prédiction du modèle ? En effet, puisque changer les hyperparamètres ne permet pas d'obtenir des résultats plus satisfaisants et que nous avons constaté que même notre meilleur modèle produit de nouvelles descriptions assez éloignées des résultats attendus, on voudrait trouver d'autres moyens pour obtenir un modèle plus performant qui puisse véritablement être utilisé dans un contexte réel (en particulier pour les malvoyants).

- La première chose à faire serait d'augmenter la durée d'entraînement (en termes d'epochs) : même si la loss de validation semble déjà avoir convergé en 40 epochs, tant qu'elle ne remonte pas on a tout intérêt à poursuivre l'apprentissage de la base d'entraînement (qui lui n'avait pas convergé) pour peut-être gagner des résultats un peu meilleurs même s'ils ne correspondent pas exactement aux descriptions déjà écrites.

- Changer ou approfondir le dataset utilisé pour l'entraînement. Le Dataset choisi n'est pas idéal car comme nous l'avons montré certains objets ou mots sont beaucoup plus récurrents que d'autres (man plus que woman ; dog plus que tout autre animal...) ce qui biaise le modèle et limite sa capacité de généralisation. Nous pouvons soit y ajouter de nouvelles images et descriptions pour espérer corriger les biais repérés par l'étude des occurences fréquentes, soit changer complètement de Dataset. Par exemple, si nous avions été amené à poursuivre ce projet nous aurions souhaité appliquer notre modèle au dataset https://huggingface.co/datasets/yuvalkirstain/pexel_images_lots_with_generated_captions qui se spécialise sur des portraits humains ou d'animaux. Cela nous aurait permis de juger de la qualité de notre premier dataset, et de voir les performances de notre modèle lorsqu'il est entraîné dans un contexte précis (ici identifier l'individu et l'activité dans un portrait).

- Nous aurions également pu produire une nouvelle tête plus riche au modèle convolutionnel pré-entraîné. Ici on s'en est servi uniquement pour lier inceptionV3 au LSTM, mais nous aurions pu ajouter quelques hidden layers fully connected pour produire des embeddings potentiellement meilleurs, ou même créer une nouvelle couche de convolution pour obtenir des feature maps plus pertinentes en premier lieu pour notre tâche.

- Enfin, le choix le plus intéressant que nous souhaiterions mettre en place est de remplacer le LSTM par un transformer de type Decoder (famille BERT...). En effet, les LSTM ont été abandonnés il y a plusieurs années pour le NLP car leurs résultats sont nettement inférieurs à ceux des transformers basés sur l'attention. La courbe suivante précise justement à quel point la précision des transformers est aujourd'hui meilleure.

<figure><center>
<img src="./Autres figures rapport/Transformer-based-VS-LSTM-based-models-performance-comparison-with-different.png">
</center></figure>

## Répartition des tâches

Même si cet historique GIT semble indiquer que nous avons chacun réalisé différentes tâches à différentes périodes lors de ce projet, la vérité est que nous avons travaillé la plupart du temps ensemble sur une machine de l'université car nos ordinateurs personnels n'étaient pas assez puissants.

En termes de répartition du travail, il serait donc plus juste de dire que Violette Pelgrims a travaillé en particulier sur le pre-processing (jusqu'à la création des dataloaders) et les tests tandis que Lucas Kloubert a davantage travaillé sur la conception du modèle et les fonctions d'entraînement et de validation. Les éléments restants (tracé des courbes, formation du dataset...) ont toujours été réalisés avec l'ensemble des membres présents.

## Références

- https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/more_advanced/image_captioning
- https://cloud.google.com/tpu/docs/inception-v3-advanced?hl=fr
- https://www.researchgate.net/profile/Han-Jiang-24/publication/327027035/figure/fig5/AS:675154577350660@1537980803771
- https://openaccess.thecvf.com/content_WACVW_2020/papers/w3/Zhang_Impact_of_ImageNet_Model_Selection_on_Domain_Adaptation_WACVW_2020_paper.pdf
- https://huggingface.co/datasets/yuvalkirstain/pexel_images_lots_with_generated_captions

### Besoin d'extraire les features de l'image

Création de notre propre CNN pour extraire les features de l'image et fournir du texte à un modèle NLP.

### Choix du modèle de NLP

Choix d'un GPT plutôt qu'un BERT car pas de nécessité de classification.

## Avancée du projet

### Samedi 2 décembre :

Mise en marche du projet. Début du travail sur notre CNN.

### Lundi 4 décembre :

Choix de plutôt s'orienter vers un CNN de visual encoding suivi par un LSTM pour le captionning.
L'idée sera d'enlever la dernière étape du CNN pour y ajouter ce qui nous intéresse et ensuite le brancher sur le LSTM pour obtenir la description de l'image.

Pour le début du projet, nous nous basons sur cette vidéo. Nous verrons ensuite pour remplacer le LSTM par un GPT.

https://www.youtube.com/watch?v=y2BaTt1fxJU

### Vendredi 8 décembre

Tentative d'implémentation de l'exemple de la vidéo. Encore quelques erreurs à corriger, pas encore push.
Pre processing des data effectué, rotation de certaines images et élimination des caption générés automatiquement ne faisant pas sens. (kettle kettle kettle kettle kettle kettle kettle kettle kettle kettle kettle kettle)

</div>
