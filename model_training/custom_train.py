import sys

sys.path.insert(0, '../modules')

from comic_dataset import *
from training_loop import *
from plot_results import *
    
if torch.cuda.is_available():
    torch.cuda.set_device(CUDA_DEVICE)
elif torch.backends.mps.is_available():
    mps_device = torch.device("mps")

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((500, 500)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((500, 500)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
}

if __name__ == '__main__':

    multiprocessing.set_start_method('spawn')

    print('Started making datasets!')

    train_processed_file = '../datasets/custom_processed_train.pth'
    val_processed_file = '../datasets/custom_processed_val.pth'

    dsets = {}

    dsets['train'] = ComicDataset(IMAGES_DIR, TRAIN_JSON, train_processed_file, data_transforms['train'])
    dsets['val'] = ComicDataset(IMAGES_DIR, VAL_JSON, val_processed_file, data_transforms['val'])

    dset_sizes = {split: len(dsets[split]) for split in ['train', 'val']}

    print('Finished making datasets!')

    dset_loaders = {}
    for split in ['train', 'val']:
        dset_loaders[split] = torch.utils.data.DataLoader(dsets[split], batch_size=BATCH_SIZE, shuffle=True, num_workers=12)

    # Larger initial kernel than ResNet50 to capture larger features, denser fully connected layers
    
    class CustomCNN(nn.Module):
        def __init__(self, num_classes):
            super(CustomCNN, self).__init__()
            # Initial large kernel convolution
            self.conv1 = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            
            # Additional convolutional layers
            self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2)
            self.bn2 = nn.BatchNorm2d(128)
            self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
            self.bn3 = nn.BatchNorm2d(256)
            self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
            self.bn4 = nn.BatchNorm2d(512)
            self.conv5 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
            self.bn5 = nn.BatchNorm2d(512)
            self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            
            # Adjusted final pooling to correct size reduction
            self.finalpool = nn.MaxPool2d(kernel_size=3, stride=3)  # Adjust to get from 25x25 to 7x7
            
            # Fully connected layers
            self.fc1 = nn.Linear(51200, 4096)
            self.fc2 = nn.Linear(4096, 1024)
            self.fc3 = nn.Linear(1024, num_classes)
            
        def forward(self, x):
            x = self.relu(self.bn1(self.conv1(x)))
            x = self.maxpool1(x)
            x = self.relu(self.bn2(self.conv2(x)))
            x = self.relu(self.bn3(self.conv3(x)))
            x = self.relu(self.bn4(self.conv4(x)))
            x = self.relu(self.bn5(self.conv5(x)))
            x = self.maxpool2(x)
            x = self.finalpool(x)
            
            x = torch.flatten(x, 1)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    model = CustomCNN(num_classes=NUM_TAGS)
    criterion = nn.BCEWithLogitsLoss()

    # Set up the device (CUDA, MPS, or CPU) based on availability and move the model and loss function to that device.
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    criterion.to(device)

    optimizer = optim.Adam(model.parameters(), lr = 0.001)

    model, accuracies, losses, hams = train_model(model, criterion, optimizer, exp_lr_scheduler, dsets, dset_sizes, dset_loaders, num_epochs=NUM_EPOCHS)

    # Output the accuracies and losses for each training epoch for both training and validation datasets.
    for split in ['train', 'val']:
        # Print the accuracies recorded during each epoch of the training and validation phases.
        print(split, 'accuracies by epoch:', accuracies[split])
        # Print the losses recorded during each epoch of the training and validation phases.
        print(split, 'losses by epoch:', losses[split])

    plot_input_data(dsets['train'], 'Custom')
    plot_training_history(accuracies, losses, hams, 'Custom')
    plot_metrics(model, dset_loaders['val'], dset_sizes['val'], 'Custom')
    torch.save(model.state_dict(), '../../models/Custom_best_model.pt')