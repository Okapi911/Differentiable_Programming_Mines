import torch
import torch.nn as nn
import torchvision.models as models

class EncoderCNN(nn.Module):
    def __init__(self, embed_size, train_CNN=False):
        super(EncoderCNN, self).__init__()
        self.inception = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3',pretrained=True) #Here choose pretrained model, if possible already without last layer
        self.last_layer = nn.Sequential(nn.Linear(self.inception.fc.in_features, embed_size), nn.ReLU(), nn.Dropout(0.5)) #We add a last layer so that we have embeddings of the specified size
        self.inception.fc = self.last_layer
        for name, param in self.inception.named_parameters():
            if "fc.weight" in name or "fc.bias" in name:
                param.requires_grad = True
            else:
                param.requires_grad = train_CNN

    def forward(self, images_batch):
        features = self.inception(images_batch)
        return features
    
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True) #batch_first=True means that the input and output tensors are provided as (batch, seq, feature)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, features_batch, captions_batch):
        embeddings = self.dropout(self.embed(captions_batch))
        embeddings = torch.cat((features_batch.unsqueeze(1), embeddings), 1) #We add the features as the first element of the sequence
        lstm_out, _ = self.lstm(embeddings)
        outputs = self.linear(lstm_out)
        return outputs
