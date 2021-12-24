
from dataset import LineFollowerDataset
from sim import Action
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision.models import resnet18
from torchvision import transforms
from tqdm import tqdm
import time # I added this
import copy # I added this

# I added this method-- large portions of the code are utilized from Pytorch tutorial for transfer learning:
# https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
def train_model(model, criterion, optimizer, dataloaders, dataset_sizes, num_epochs=30):
    start_time = time.time()

    best_model = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        start_epoch = time.time()
        print('Epoch {}/{}'.format(epoch+1, num_epochs))

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            total_loss = 0.0
            total_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device) # putting inputs and labels onto cuda or cpu
                labels = labels.to(device)

                optimizer.zero_grad() # zero the parameter gradients

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    #print("outputs: ", outputs)
                    #print("labels: ", labels)
                    loss = criterion(outputs, labels)
                    #loss = criterion(outputs, labels.unsqueeze(1).float())

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                total_loss += loss.item() * inputs.size(0) # statistics
                total_corrects += torch.sum(preds == labels.data)

            epoch_loss = total_loss / dataset_sizes[phase]
            epoch_acc = total_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model = copy.deepcopy(model.state_dict())

        time_epoch = time.time() - start_epoch
        print("Time for epoch: ", time_epoch)
        print()

    time_elapsed = time.time() - start_time
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model)
    return model

def save_chkpt(model, path): # I added this method
    #copy.deepcopy(model.state_dict())?
    torch.save(model.state_dict(), path)

def load_chkpt(model, path): # I added this method
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(path, map_location=device))
    return model


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # TODO: Load the dataset
    # - You might want to use a different transform than what is already provided
    #dataset = LineFollowerDataset(transform=transforms.Compose([
    #    transforms.ToTensor()]))
    dataset = LineFollowerDataset(transform=transforms.ToTensor())

    # TODO: Prepare dataloaders
    # - Rnadomly split the dataset into the train validation dataset.
    # 	* Hint: Checkout torch.utils.data.random_split
    # - Prepare train validation dataloaders
    # ========================================
    len_dataset= len(dataset)
    b_size= 5 # batch_size
    train_set, val_set= torch.utils.data.random_split(dataset, [round(0.85*len_dataset), round(0.15*len_dataset)])
    train_loader= DataLoader(train_set, batch_size=b_size, shuffle=True, num_workers=2) # creates an iterator from which to get training samples (batch_size=100)
    val_loader= DataLoader(val_set, batch_size=b_size, shuffle=True, num_workers=2)
    dataloaders = {'train': train_loader, 'val': val_loader}
    dataset_sizes = {'train': len(train_set), 'val': len(val_set)}
    # ========================================

    # TODO: Prepare model
    # - You might want to use a pretrained model like resnet18 and finetune it for your dataset
    # - See https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
    # ========================================
    model = resnet18(pretrained=True)

    # freezing layers
    """for param in model.parameters():
        param.requires_grad = False"""

    # replacing last fc layer
    num_ftrs = model.fc.in_features
    num_classes= 4

    top_model= nn.Sequential(
            nn.Linear(num_ftrs, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes), # change to 2 or 1
            nn.ReLU(inplace=True))

    model.fc = top_model
    print("model: ", model)
    # ========================================

    # TODO: Prepare loss and optimizer
    # ========================================
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    # ========================================

    # TODO: Train model
    # - You might want to print training (and validation) loss and accuracy every epoch
    # - You might want to save your trained model every epoch
    # ========================================
    model = train_model(model, criterion, optimizer, dataloaders, dataset_sizes, num_epochs=5)
    path= 'following_model.pth'
    save_chkpt(model, path) # save best model checkpoint
    # ========================================
