import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
import os
from Image_Captioning import CNN_TO_RNN
from few_examples_results import print_examples
import torch.optim as optim
from get_loader import get_loader
from torch.utils.tensorboard import SummaryWriter
from Image_Captioning import CNN_TO_RNN
import matplotlib.pyplot as plt
from eval_model import eval_CNN_to_RNN
from dataset_getter import get_dataset
import matplotlib.image as mpimg


valid_loss = torch.load('valid_lossf.pt')
plt.plot(valid_loss)
plt.show()

losses = torch.load('lossesf.pt')
plt.plot(losses)
plt.show()



losses = torch.load('losses_per_epochf.pt')
plt.plot(losses)
plt.show()


embed_size = 256
hidden_size = 256
num_layers = 1
learning_rate = 3e-4
num_epochs = 5
batch_size = 32



transform = transforms.Compose(
        [
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )


_, _, test_loader, dataset = get_loader(
        root_folder="./flickr8k/images",
        annotation_file="./flickr8k/captions.txt",
        transform=transform,
        num_workers=2,
    )

"""
dataset = get_dataset(
        root_folder="./flickr8k/images",
        annotation_file="./flickr8k/captions.txt",
        transform=transform,
        num_workers=2)
"""


vocab_size = len(dataset.vocab)


model = CNN_TO_RNN(embed_size, hidden_size, vocab_size, num_layers)
model.load_state_dict(torch.load('model_image_captioningf.pt'))
#model = torch.load('model_image_captioning.pt')

test_loss = eval_CNN_to_RNN('model_image_captioning3.pt', test_loader, loss_fn=nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi["<PAD>"]), embed_size=embed_size, hidden_size=hidden_size, vocab_size=vocab_size, num_layers=num_layers)
print("The value of the loss on the test dataset is t_loss = " + str(test_loss))


model.eval()

imgs_path = []
captions = []

path = "Examples/dog.jpg"
imgs_path.append(path)
test_img1 = transform(Image.open(path).convert("RGB")).unsqueeze(
        0
    )
caption = " ".join(model.caption_image(test_img1, dataset.vocab))
captions.append(caption)
print("Example 1: Dog on a beach by the ocean")
print(
        "Example 1 OUTPUT: "
        + caption
    )

path = "Examples/boat.png"
imgs_path.append(path)
test_img2 = transform(Image.open(path).convert("RGB")).unsqueeze(
        0
    )
caption = " ".join(model.caption_image(test_img2, dataset.vocab))
captions.append(caption)
print("Example 2: A small boat in the ocean")
print(
        "Example 2 OUTPUT: "
        + caption
    )


path = "Examples/bus.png"
imgs_path.append(path)
test_img3 = transform(Image.open(path).convert("RGB")).unsqueeze(
        0
    )
caption = " ".join(model.caption_image(test_img3, dataset.vocab))
captions.append(caption)
print("Example 3: Bus driving by parked cars")
print(
        "Example 3 OUTPUT: "
        + caption
    )


path = "Examples/child.jpg"
imgs_path.append(path)
test_img4 = transform(Image.open(path).convert("RGB")).unsqueeze(
        0
    )
caption = " ".join(model.caption_image(test_img4, dataset.vocab))
captions.append(caption)
print("Example 4: Child holding red frisbee outdoors")
print(
        "Example 4 OUTPUT: "
        + caption
    )


path = "Examples/horse.png"
imgs_path.append(path)
test_img5 = transform(Image.open(path).convert("RGB")).unsqueeze(
        0
    )
caption = " ".join(model.caption_image(test_img5, dataset.vocab))
captions.append(caption)
print("Example 5: A cowboy riding a horse in the desert")
print(
        "Example 5 OUTPUT: "
        + caption
    )

"""
To print a test batch

for idx, (imgs, captions) in tqdm(
            enumerate(test_loader), total=len(test_loader), leave=False
        ):
    plt.figure(figsize=(10,6))
    for ib in range (8):
        plt.subplot(8 // 4, 4, ib+1)
        plt.imshow(imgs[ib,:].squeeze().detach().permute(1,2,0))
        plt.xticks([]), plt.yticks([])
    plt.show()
    break
"""




plt.figure(figsize=(15,10))
for ib in range(4):
    plt.subplot(4 // 1, 1, ib+1)
    #tight_layout(pad=5.0)
    plt.imshow(mpimg.imread(imgs_path[ib]))
    plt.title(captions[ib])
    plt.xticks([]), plt.yticks([])
plt.show()
    

