import random
from datasets import load_dataset
from torch.utils.data import random_split
from torch.utils.data import DataLoader

dataset = load_dataset(
    "yuvalkirstain/pexel_images_lots_with_generated_captions")

# We prepare our data to have some rotations and to be all RGB. Each image is already 512x512 so this is already done.

for i in range(len(dataset['train'])):
    if i % 100 == 0:
        image = dataset['train'][i]['image'].convert('RGB')
        angle = random.choice([90, 180, 270])
        # Rotate the image
        dataset['train'][i]['image'] = image.rotate(angle)
    else:
        dataset['train'][i]['image'] = dataset['train'][i]['image'].convert(
            'RGB')

# We noticed than some caption seem to have been generated automaticaly and were very long and weird so we delete those.
dataset['train'] = [data for data in dataset['train']
                    if len(data['generated_caption']) <= 101]


# Split dataset into train, validation, and test sets
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

num_samples = len(dataset['train'])
train_size = int(train_ratio * num_samples)
val_size = int(val_ratio * num_samples)
test_size = num_samples - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    dataset['train'], [train_size, val_size, test_size])

# Define data loaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)
