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
from eval_model import eval_CNN_to_RNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using {} device".format(device))

def train(batch_size):
    transform = transforms.Compose(
        [
            transforms.Resize((356, 356)),
            transforms.RandomCrop((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    
    train_loader, valid_train, test_loader, dataset = get_loader(
        root_folder="./flickr8k/images",
        annotation_file="./flickr8k/captions.txt",
        transform=transform,
        num_workers=2,
        batch_size = batch_size,
        type = 'train'
    )

    torch.backends.cudnn.benchmark = True
    load_model = False
    save_model = False
    train_CNN = True

    # Hyperparameters
    embed_size = 256
    hidden_size = 256
    vocab_size = len(dataset.vocab)
    num_layers = 1
    learning_rate = 3e-4
    num_epochs = 1


    # for tensorboard
    writer = SummaryWriter("runs/flickr")
    step = 0

    # initialize model, loss etc
    model = CNN_TO_RNN(embed_size, hidden_size, vocab_size, num_layers).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi["<PAD>"])
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Only finetune the CNN
    for name, param in model.encoder.inception.named_parameters():
        if "fc.weight" in name or "fc.bias" in name:
            param.requires_grad = True
        else:
            param.requires_grad = train_CNN

    if load_model:
        step = model.load_state_dict(torch.load("model_image_captioning.pt"))#load_checkpoint(torch.load("my_checkpoint.pth.tar"), model, optimizer)

    model.train()

    losses = []
    losses_per_epoch = []
    valid_losses = []

    for epoch in range(num_epochs):

        loss_this_epoch = 0.

        if save_model:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
            }
            torch.save(checkpoint, "model_image_captioning.pt")

        for idx, (imgs, captions) in tqdm(
            enumerate(train_loader), total=len(train_loader), leave=False
        ):
            imgs = imgs.to(device)
            captions = captions.to(device)

            outputs = model(imgs, captions[:-1])
            loss = criterion(
                outputs.reshape(-1, outputs.shape[2]), captions.reshape(-1)
            )

            losses.append(loss.item())
            loss_this_epoch += loss.item()

            writer.add_scalar("Training loss", loss.item(), global_step=step)
            step += 1

            optimizer.zero_grad()
            loss.backward(loss)
            optimizer.step()
        
        loss_this_epoch = loss_this_epoch/(len(train_loader)*batch_size)
        losses_per_epoch.append(loss_this_epoch)

        torch.save(model.state_dict(), 'model_temp.pt')

        valid_loss = eval_CNN_to_RNN('model_temp.pt', train_loader, loss_fn=nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi["<PAD>"]), embed_size=embed_size, hidden_size=hidden_size, vocab_size=vocab_size, num_layers=num_layers)
        valid_losses.append(valid_loss)

    return model, train_loader, dataset, losses, losses_per_epoch, valid_loss


if __name__ == "__main__":
    batch_size=32
    model, train_loader, dataset, losses, losses_per_epoch, valid_loss = train(batch_size)
    print('finished normally')
    torch.save(model.state_dict(), 'model_image_captioning3.pt')

    import matplotlib.pyplot as plt

    plt.plot(losses)
    plt.show()

    torch.save(losses, 'losses3.pt')

    plt.plot(losses_per_epoch)
    plt.plot(valid_loss)
    plt.show()

    torch.save(losses_per_epoch, 'losses_per_epoch3.pt')
    torch.save(valid_loss, 'valid_loss3.pt')

    #Tests

    #To change
    test_loader = train_loader
    
    """
    score_test = eval_CNN_to_RNN(model, test_loader, nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi["<PAD>"]))

    print('test loss =' + str(score_test))
    torch.save(score_test, 'losses_test3.pt')
    
    image_batch_example, labels_batch_example = next(iter(test_loader))
    plt.figure(figsize=(10,6))
    for ib in range (batch_size):
        plt.subplot(batch_size // 4, 4, ib+1)
        plt.imshow(image_batch_example[ib,:].squeeze().detach().permute(1,2,0))
        plt.xticks([]), plt.yticks([])
        plt.title('Image caption = ' + str(labels_batch_example[ib].item()))
    
    """
