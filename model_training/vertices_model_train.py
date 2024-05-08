import sys

sys.path.insert(0, '../modules')

from vertices_dataset import *
from training_loop import *
from plot_results import *
    
if torch.cuda.is_available():
    torch.cuda.set_device(CUDA_DEVICE)
elif torch.backends.mps.is_available():
    mps_device = torch.device("mps")

if __name__ == '__main__':

    multiprocessing.set_start_method('spawn')

    print('Started making datasets!')

    train_processed_file = '../datasets/vertices_processed_train.pth'
    val_processed_file = '../datasets/vertices_processed_val.pth'

    dsets = {}

    dsets['train'] = VerticesDataset(IMAGES_DIR, TRAIN_JSON, train_processed_file)
    dsets['val'] = VerticesDataset(IMAGES_DIR, VAL_JSON, val_processed_file)

    dset_sizes = {split: len(dsets[split]) for split in ['train', 'val']}

    print('Finished making datasets!')

    dset_loaders = {}
    for split in ['train', 'val']:
        dset_loaders[split] = torch.utils.data.DataLoader(dsets[split], batch_size=BATCH_SIZE, shuffle=True, num_workers=12)

    class CustomVertexCNN(nn.Module):
        def __init__(self, num_classes):
            super(CustomVertexCNN, self).__init__()
            
            # Using 1D convolutions with a kernel size of 4
            self.conv1 = nn.Conv1d(1, 64, kernel_size=4, stride=1, padding=1)
            self.bn1 = nn.BatchNorm1d(64)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
            
            self.conv2 = nn.Conv1d(64, 128, kernel_size=4, stride=1, padding=1)
            self.bn2 = nn.BatchNorm1d(128)
            self.conv3 = nn.Conv1d(128, 256, kernel_size=4, stride=1, padding=1)
            self.bn3 = nn.BatchNorm1d(256)
            self.conv4 = nn.Conv1d(256, 512, kernel_size=4, stride=1, padding=1)
            self.bn4 = nn.BatchNorm1d(512)
            self.conv5 = nn.Conv1d(512, 512, kernel_size=4, stride=1, padding=1)
            self.bn5 = nn.BatchNorm1d(512)
            self.maxpool2 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
            
            # Fully connected layers for classification
            self.fc1 = nn.Linear(4096, 2048)  # Adjust the input features according to your feature map size before this layer
            self.fc2 = nn.Linear(2048, 1024)
            self.fc3 = nn.Linear(1024, num_classes)
            
        def forward(self, x):
            x = x.unsqueeze(1)  # Adding a channel dimension, assuming x has shape [batch_size, num_features]
            x = self.relu(self.bn1(self.conv1(x)))
            x = self.maxpool1(x)
            x = self.relu(self.bn2(self.conv2(x)))
            x = self.relu(self.bn3(self.conv3(x)))
            x = self.relu(self.bn4(self.conv4(x)))
            x = self.relu(self.bn5(self.conv5(x)))
            x = self.maxpool2(x)
            
            x = torch.flatten(x, 1)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = torch.sigmoid(self.fc3(x))  # Sigmoid activation for multi-label classification
            return x

    # Model instantiation
    model = CustomVertexCNN(num_classes=NUM_TAGS)
    criterion = nn.BCELoss()  # Appropriate loss for binary classification tasks

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

    plot_training_history(accuracies, losses, hams, 'Vertices')
    plot_metrics(model, dset_loaders['val'], dset_sizes['val'], 'Vertices')
    torch.save(model.state_dict(), '../../models/Vertices_best_model.pt')