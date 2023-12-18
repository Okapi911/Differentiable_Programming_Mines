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



def eval_CNN_to_RNN(path, eval_dataloader, loss_fn, embed_size, hidden_size, vocab_size, num_layers):
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN_TO_RNN(embed_size, hidden_size, vocab_size, num_layers)
    model.load_state_dict(torch.load(path))
    model.train()

    with torch.no_grad():
        valid_loss = 0

        for idx, (images, captions) in tqdm(enumerate(eval_dataloader), total=len(eval_dataloader), leave=False):
        #for images, captions in eval_dataloader:

            #images = images.to(device)
            #captions = captions.to(device)
            #print(images)
            #print(images.shape)
            #print(captions.shape)
            #print(captions[:-1].shape)
            batch_predicted = model(images, captions[:-1])
            

            loss_this_batch = loss_fn(
                batch_predicted.reshape(-1, batch_predicted.shape[2]), captions.reshape(-1)
            )
            valid_loss += loss_this_batch.item() #* images.shape[0]
        
    valid_loss = valid_loss / len(eval_dataloader.dataset)
    return valid_loss