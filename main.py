import os
from constants import *
import torch
import multiprocessing
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset
from PIL import Image
from PIL import ImageFile
import os
import requests
import json 
from tqdm import tqdm
import time
from torch.autograd import Variable
ImageFile.LOAD_TRUNCATED_IMAGES = True
import copy
import numpy as np
import matplotlib.pyplot as plt

# Copied from custom_hymenoptera_dataset.py
# Checks for valid image files (size and extension)
def is_valid_image_file(filename, max_pixels=178956970):
    # Check file name extension
    valid_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']
    if os.path.splitext(filename)[1].lower() not in valid_extensions:
        print(f"Invalid image file extension \"{filename}\". Skipping this file...")
        return False
    
    # Temporarily disable the decompression bomb check
    original_max_image_pixels = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = None
    
    # Verify that image file is intact and check its size
    try:
        with Image.open(filename) as img:
            img.verify()  # Verify if it's an image
            # Restore the original MAX_IMAGE_PIXELS limit
            Image.MAX_IMAGE_PIXELS = original_max_image_pixels
            
            # Check image size without loading the image into memory
            if img.size[0] * img.size[1] > max_pixels:
                print(f"Image {filename} is too large. Skipping this file...")
                return False
            return True
    except (IOError, SyntaxError) as e:
        print(f"Invalid image file {filename}: {e}")
        # Ensure the MAX_IMAGE_PIXELS limit is restored even if an exception occurs
        Image.MAX_IMAGE_PIXELS = original_max_image_pixels
        return False
    # Ensure the MAX_IMAGE_PIXELS limit is restored in case of any other unexpected exit
    Image.MAX_IMAGE_PIXELS = original_max_image_pixels

class ComicDataset(Dataset):
    # Hot encode our labels for our targets
    def hot_encode_target(self, tags):
        target = torch.zeros(NUM_TAGS)
        for tag in tags:
            target[TAGS.index(tag)] = 1

        return target

    def __init__(self, images_dir, json_file, transform=None, target_transform=None):
        self.images_dir = images_dir
        self.transform = transform
        self.target_transform = target_transform

        image_label_dict = {}
        class_counts = {}

        for tag in TAGS:
            class_counts[tag] = 0

        with open(json_file, 'r') as f:
            data = json.load(f)

        progress_bar = tqdm(total=len(data.items()))
        for key, value in data.items():
            if key in os.listdir(images_dir):
                if is_valid_image_file(os.path.join(self.images_dir, key)):
                    target = self.hot_encode_target(value)

                    if len(target) == NUM_TAGS:
                        image_label_dict[key] = target
                        for v in value:
                            class_counts[v] += 1

                    else:
                        print('Invalid file: ' + key + '. Skipping this file...')
            progress_bar.update(1)

        self.items = list(image_label_dict.items())
        print('Class counts: ', class_counts)

        if (sum(class_counts.values()) > 23000):
            phase = "TRAIN"
        else:
            phase = "VAL"
        print(f"{phase.upper()} SET STATISTICS:")
        total_images = sum(class_counts.values())
        print(f"Total images: {total_images}")
        for class_id, count in class_counts.items():
            print(f"Class {class_id}: {count} images")
        print("\n")

    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.items[idx][0])
        image = Image.open(img_path).convert('RGB')  # Convert image to RGB
    
        label = self.items[idx][1]

        if self.transform:
            image = self.transform(image)
            if image.shape[0] == 1:
                new_image = torch.zeros((3, 224, 224))
                for i in range(3):
                    new_image[i] = image
                
                image = new_image
        if self.target_transform:
            label = self.target_transform(label)
        
        return image, label

    
use_gpu = GPU_MODE
use_mps = MPS_MODE
if use_gpu:
    torch.cuda.set_device(CUDA_DEVICE)

if use_mps:
   mps_device = torch.device("mps")

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
}

def train_model(model, criterion, optimizer, lr_scheduler, num_epochs=100):
    since = time.time()

    best_model = model
    best_loss = torch.inf

    losses = {'train': [], 'val': []}

    for epoch in range(num_epochs):
        print('-' * 10)
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                optimizer = lr_scheduler(optimizer, epoch)
                model.train()
            else:
                model.eval()
        
            running_loss = 0.0

            counter = 0

            progress_bar = tqdm(total=dsets[phase].__len__())
            for data in dset_loaders[phase]:
                inputs, labels = data

                if use_gpu:
                    try:
                        inputs, labels = Variable(inputs.float().cuda()), Variable(labels.long().cuda())

                    except Exception as e:
                        print("ERROR! here are the inputs and labels before we print the full stack trace:")
                        print(inputs, labels)
                        raise e
                    
                elif use_mps:
                   try:
                      inputs, labels = Variable(inputs.float().to(mps_device)), Variable(labels.long().to(mps_device))

                   except Exception as e:
                      print("ERROR! here are the inputs and labels before we print the full stack trace:")
                      print(inputs, labels)
                      raise e
                
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                
                optimizer.zero_grad()
                outputs = model(inputs)

                loss = criterion(outputs, labels.float())

                if counter%100 == 0:
                    print('Reached batch iteration', counter)

                counter += 1
                progress_bar.update(BATCH_SIZE)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                try:
                    running_loss += loss.item()
                except:
                    print('unexpected error, could not calculate loss or do a sum.')

            epoch_loss = running_loss / dset_sizes[phase]
            print('{} Loss: {:.4f}'.format(phase, epoch_loss))
            losses[phase].append(epoch_loss)

            # deep copy the model
            if phase == 'val':
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model = copy.deepcopy(model)
                    print('new best loss =', best_loss)
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))
    print('returning and looping back')

    return best_model, losses

def exp_lr_scheduler(optimizer, epoch, init_lr=BASE_LR, lr_decay_epoch=EPOCH_DECAY):
    """Decay learning rate by a factor of DECAY_WEIGHT every lr_decay_epoch epochs."""
    lr = init_lr * (DECAY_WEIGHT**(epoch // lr_decay_epoch))

    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer

if __name__ == '__main__':

    multiprocessing.set_start_method('spawn')

    print('Started making datasets!')

    dsets = {}

    dsets['train'] = ComicDataset(IMAGES_DIR, TRAIN_JSON, data_transforms['train'])
    dsets['val'] = ComicDataset(IMAGES_DIR, VAL_JSON, data_transforms['val'])

    dset_sizes = {split: len(dsets[split]) for split in ['train', 'val']}

    print('Finished making datasets!')

    dset_loaders = {}
    for split in ['train', 'val']:
        dset_loaders[split] = torch.utils.data.DataLoader(dsets[split], batch_size=BATCH_SIZE, shuffle=True, num_workers=12)

    class ResNet50MultiLabel(nn.Module):
        def __init__(self, num_classes):
            super(ResNet50MultiLabel, self).__init__()
            # Load a pre-trained ResNet-50 model
            self.resnet50 = models.resnet50(pretrained=True)
            
            # Replace the final fully connected layer
            # ResNet-50's fc layer output features is 2048
            self.resnet50.fc = nn.Linear(2048, num_classes)
        
        def forward(self, x):
            return self.resnet50(x)

    model = ResNet50MultiLabel(NUM_TAGS)
    criterion = nn.BCEWithLogitsLoss()

    if use_gpu:
        criterion.cuda()
        model.cuda()

    if use_mps:
        criterion.to(mps_device)
        model.to(mps_device)

    optimizer = optim.Adam(model.parameters(), lr = 0.001)

    model, losses = train_model(model, criterion, optimizer, exp_lr_scheduler, num_epochs=NUM_EPOCHS)

    for split in ['train', 'val']:
        print(split, 'losses by epoch:', losses[split])

    figure = plt.figure(figsize=(8, 8))
    cols, rows = 3, 3
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(dsets['train']), size=(1,)).item()
        img, label = dsets['train'][sample_idx]

        # Convert the tensor image to numpy
        img = img.numpy().transpose((1, 2, 0))  # Change from (C, H, W) to (H, W, C)
        
        # Undo the normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean  # Apply the inverse of the initial normalization
        img = np.clip(img, 0, 1)  # Ensure the values are between 0 and 1

        # Plot the image
        figure.add_subplot(rows, cols, i)

        title = ''
        for i in range(NUM_TAGS):
            if label[i] == 1:
                title += (TAGS[i] + ' ')

        plt.title(title)
        plt.axis("off")
        plt.imshow(img)  # img is now in the correct format for imshow

    plt.savefig('train_images.png')
    plt.show()


    def plot_training_history(losses):
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 2)
        for phase in ['train', 'val']:
            plt.plot(losses[phase], label=f'{phase} loss')
        plt.title('Loss over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.savefig('loss_plot.png')
        plt.show()

    plot_training_history(losses)
    torch.save(model.state_dict(), 'fine_tuned_best_model.pt')