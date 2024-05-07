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
        transforms.Resize((600, 600)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((600, 600)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
}

if __name__ == '__main__':

    multiprocessing.set_start_method('spawn')

    print('Started making datasets!')

    train_processed_file = '../datasets/EffNetB0_processed_train.pth'
    val_processed_file = '../datasets/EffNetB0_processed_val.pth'

    dsets = {}

    dsets['train'] = ComicDataset(IMAGES_DIR, TRAIN_JSON, train_processed_file, data_transforms['train'])
    dsets['val'] = ComicDataset(IMAGES_DIR, VAL_JSON, val_processed_file, data_transforms['val'])

    dset_sizes = {split: len(dsets[split]) for split in ['train', 'val']}

    print('Finished making datasets!')

    dset_loaders = {}
    for split in ['train', 'val']:
        dset_loaders[split] = torch.utils.data.DataLoader(dsets[split], batch_size=BATCH_SIZE, shuffle=True, num_workers=12)

    # Larger initial kernel than ResNet50 to capture larger features, denser fully connected layers
    class EfficientNetB0MultiLabel(nn.Module):
        def __init__(self, num_classes):
            super(EfficientNetB0MultiLabel, self).__init__()
            # Load a pre-trained EfficientNet-B0 model
            self.efficientnet_b0 = timm.create_model('efficientnet_b0', pretrained=True)
            
            # Replace the final fully connected layer
            # EfficientNet-B0's classifier layer output features is 1280
            self.efficientnet_b0.classifier = nn.Linear(1280, num_classes)
        
        def forward(self, x):
            return self.efficientnet_b0(x)

    model = EfficientNetB0MultiLabel(NUM_TAGS)
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

    plot_input_data(dsets['train'], 'EffNetB0')
    plot_training_history(accuracies, losses, hams, 'EffNetB0')
    plot_metrics(model, dset_loaders['val'], dset_sizes['val'], 'EffNetB0')
    torch.save(model.state_dict(), '../../models/EffNetB0_best_model.pt')