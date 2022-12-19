"""# Imports"""
import numpy as np 
import matplotlib.pyplot as plt
import torch
"""# Configurations"""

# This are the configurations

TRAIN_IMG_FOLDER_PATH = '/content/Facial-Expression-Dataset/train'
VALID_IMG_FOLDER_PATH = '/content/Facial-Expression-Dataset/validation'

lr = 0.001
bath_size = 32
epochs = 15

device = 'cuda'
model_name = 'efficientnet_b0'

"""# Load Dataset """

from torchvision.datasets import ImageFolder
from torchvision import transforms as T

from torchvision.transforms.transforms import ToTensor
# dynamic augmentation of the images. Since the augmentation is 
#added at training step the size of the initial input does not increase
train_augs = T.Compose([
    T.RandomHorizontalFlip(0.5),
    T.RandomRotation(degrees = (-20, +20)),
    T.ToTensor() # converts numpy or pil to torch tensor and 
    #(height, width, channel) to (channel, height, weight)
])

valid_augs = T.Compose([
    T.ToTensor()
])

trainset = ImageFolder(TRAIN_IMG_FOLDER_PATH, transform = train_augs)
validset = ImageFolder(VALID_IMG_FOLDER_PATH, transform=valid_augs)

print(f"Total no. of examples in trainset : {len(trainset)}")
print(f"Total no. of examples in validset : {len(validset)}")

print(trainset.class_to_idx)

image, label = trainset[40]
plt.imshow(image.permute(1,2,0))
plt.title(label)

"""# Load Dataset into Batches """

from torch.utils.data import DataLoader

from torchvision.datasets.folder import DatasetFolder
# Data loders 
trainloader = DataLoader(trainset, batch_size=bath_size, shuffle = True)
validloader = DataLoader(validset, batch_size=bath_size)

print(f"Total no. of batches in trainloader : {len(trainloader)}")
print(f"Total no. of batches in validloader : {len(validloader)}")

for images, labels in trainloader:
  break

print(f"One image batch shape : {images.shape}")
print(f"One label batch shape : {labels.shape}")

"""# Create Model """

import timm 
from torch import nn

class FaceModel(nn.Module):

  def __init__(self):
    super(FaceModel, self).__init__()

    self.eff_net = timm.create_model('efficientnet_b0', pretrained=True, num_classes =7)

  def forward(self, images, labels=None):
    logits = self.eff_net(images)

    if labels != None:
      loss = nn.CrossEntropyLoss()(logits, labels)
      return logits, loss
    
    return logits

model = FaceModel()
model.to(device)



"""# Create Train and Eval Function """

from tqdm import tqdm

def multiclass_accuracy(y_pred,y_true):
    top_p,top_class = y_pred.topk(1,dim = 1)
    equals = top_class == y_true.view(*top_class.shape)
    return torch.mean(equals.type(torch.FloatTensor))

def train_fn(model, dataloader, optimizer, current_epoch):

  model.train()
  total_loss = 0.0
  total_acc = 0.0
  tk = tqdm(dataloader, desc='EPOCH' + '[TRAIN]' + str(current_epoch + 1) + '/' + str(epochs))

  for t, data in enumerate(tk):
    images, labels = data
    images, labels = images.to(device), labels.to(device)

    optimizer.zero_grad()
    logits, loss  = model(images, labels)

    loss.backward()
    optimizer.step()

    total_loss += loss.item()
    total_acc += multiclass_accuracy(logits, labels)

    tk.set_postfix({'loss': '%6f' %float(total_loss / (t+1)), 'acc': '%6f' %float(total_acc / (t+1))})
  
  return total_loss / len(dataloader),  total_acc / len(dataloader)



"""# Create Training Loop"""

def eval_fn(model, dataloader, current_epoch):

  model.eval()
  total_loss = 0.0
  total_acc = 0.0
  tk = tqdm(dataloader, desc='EPOCH' + '[VALID]' + str(current_epoch + 1) + '/' + str(epochs))

  for t, data in enumerate(tk):
    images, labels = data
    images, labels = images.to(device), labels.to(device)

    logits, loss  = model(images, labels)

    total_loss += loss.item()
    total_acc += multiclass_accuracy(logits, labels)

    tk.set_postfix({'loss': '%6f' %float(total_loss / (t+1)), 'acc': '%6f' %float(total_acc / (t+1))})
  
  return total_loss / len(dataloader),  total_acc / len(dataloader)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)

best_valid_loss = np.inf
for i in range(epochs):
  train_loss, train_acc = train_fn(model, trainloader, optimizer, i)
  valid_loss, valid_acc = eval_fn(model, validloader, i)

  if valid_loss < best_valid_loss:
    best_valid_loss = valid_loss 
    torch.save(model.state_dict(), 'best-weights.pt')
    print('Saved Best Weights')

"""# Inference"""

def view_classify(img, ps):
    
    classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

    ps = ps.data.cpu().numpy().squeeze()
    img = img.numpy().transpose(1,2,0)
   
    fig, (ax1, ax2) = plt.subplots(figsize=(5,9), ncols=2)
    ax1.imshow(img)
    ax1.axis('off')
    ax2.barh(classes, ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(classes)
    ax2.set_yticklabels(classes)
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)

    plt.tight_layout()

    return None

image, label = validset[97]
image  = image.unsqueeze(0)

logits = model(image.to(device))
probs = nn.Softmax(dim=1)(logits)

view_classify(image.squeeze(), probs)

"""# For updates about upcoming and current guided projects follow me on...

Twitter : @parth_AI

Linkedin : www.linkedin.com/in/pdhameliya
"""