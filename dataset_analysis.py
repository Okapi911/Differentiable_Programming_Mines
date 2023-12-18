import re
from collections import Counter

common_words = ["a", "in", "is", "the", "and", "to", "of", "for", "on", "with", "are", "at", "from", "as", "an", "by"]

# Contenu de captions.txt
with open("../flickr8k/captions.txt", "r") as file:
    text = file.read()
text = re.sub(r'[^\w\s]', '', text.lower())
words = text.split()

words = [word for word in words if word not in common_words]
word_counts = Counter(words)
top_words = word_counts.most_common(100)
for word, count in top_words:
    print(word, count)

# Associations de 2 mots
associations_2 = [(words[i], words[i+1]) for i in range(len(words)-1)]
association_counts_2 = Counter(associations_2)
top_associations_2 = association_counts_2.most_common(100)
for association, count in top_associations_2:
    print(association, count)

# Associations de 3 mots
associations_3 = [(words[i], words[i+1], words[i+2]) for i in range(len(words)-2)]
association_counts_3 = Counter(associations_3)
top_associations_3 = association_counts_3.most_common(100)
for association, count in top_associations_3:
    print(association, count)

#Les associations de 4 mots n'ont pas été gardées car moins pertinentes et présentant beaucoup de redites
associations_4 = [(words[i], words[i+1], words[i+2], words[i+3]) for i in range(len(words)-3)]
association_counts_4 = Counter(associations_4)
top_associations_4 = association_counts_4.most_common(100)
for association, count in top_associations_4:
    print(association, count)
    