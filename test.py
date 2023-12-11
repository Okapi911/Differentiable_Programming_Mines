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

losses = torch.load('losses.pt')

import matplotlib.pyplot as plt

plt.plot(losses)
plt.show()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using {} device".format(device))

model = torch.load('model_image_captioning.pt')

transform = transforms.Compose(
        [
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

_, dataset = get_loader(
        root_folder="./flickr8k/images",
        annotation_file="./flickr8k/captions.txt",
        transform=transform,
        num_workers=2,
    )

model.eval()
test_img1 = transform(Image.open("Examples/dog.jpg").convert("RGB")).unsqueeze(
        0
    )
print("Example 1 CORRECT: Dog on a beach by the ocean")
print(
        "Example 1 OUTPUT: "
        + " ".join(model.caption_image(test_img1.to(device), dataset.vocab))
    )
