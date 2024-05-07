from constants import *

def plot_input_data(dset, model_name):
    figure = plt.figure(figsize=(8, 8))
    cols, rows = 3, 3
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(dset), size=(1,)).item()
        img, label = dset[sample_idx]

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

    plt.savefig(f'../plots/{model_name}_train_images.png')

def plot_training_history(accuracies, losses, hams, model_name):
    plt.figure(figsize=(18, 6))  # Adjust the size to accommodate three plots

    # Plot for accuracies
    plt.subplot(1, 3, 1)  # Change subplot grid to 1 row, 3 columns, and select the 1st subplot
    for phase in ['train', 'val']:
        plt.plot(accuracies[phase], label=f'{phase} accuracy')
    plt.title(f'Accuracy over Epochs for {model_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot for losses
    plt.subplot(1, 3, 2)  # Select the 2nd subplot
    for phase in ['train', 'val']:
        plt.plot(losses[phase], label=f'{phase} loss')
    plt.title(f'Loss over Epochs for {model_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot for Hamming losses
    plt.subplot(1, 3, 3)  # Select the 3rd subplot
    for phase in ['train', 'val']:
        plt.plot(hams[phase], label=f'{phase} hamming loss')
    plt.title(f'Hamming Loss over Epochs for {model_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Hamming Loss')
    plt.legend()

    plt.tight_layout()  # This adjusts subplots to give some padding between them
    plt.savefig(f'../plots/{model_name}_training_history.png')  # Save to a file

def plot_metrics(model, val_dload, val_size, model_name, device=None):
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    total_y_true = []
    total_y_pred = []

    print('creating confusion matrices:')

    progress_bar = tqdm(total=val_size, position=1)
    for inputs, labels in val_dload:
        inputs = inputs.float().to(device)  # Move inputs to device and ensure data type is float.
        labels = labels.long().to(device)  # Move labels to device and ensure data type is long.

        outputs = model(inputs)  # Get model outputs.

        y_true = labels.to('cpu').numpy()
        y_pred = outputs.to('cpu').detach().numpy()

        y_pred = (y_pred > 0).astype(int)

        total_y_true.extend(y_true)
        total_y_pred.extend(y_pred)

        progress_bar.update(BATCH_SIZE)
    
    progress_bar.close()

    confusion_matrices = multilabel_confusion_matrix(total_y_true, total_y_pred)

    # Set up the matplotlib figure and size
    plt.figure(figsize=(15, 20))  # Adjust overall figure size as needed

    cols = 3  # You can adjust the number of columns based on your preference
    rows = int(np.ceil(NUM_TAGS / cols))

    # Loop through the list of confusion matrices
    for i, matrix in enumerate(confusion_matrices):
        plt.subplot(rows, cols, i + 1)  # Arrange plots in a 4x3 grid
        sns.heatmap(matrix, annot=True, fmt='g', cmap='Blues', cbar=True)  # Plot heatmap with color bar
        
        # Optional: customize each subplot
        plt.title(f'{model_name} - Confusion Matrix for {TAGS[i]}')

    # Show the plot
    plt.tight_layout()  # Adjust layout to fit each subplot
    plt.savefig(f'../plots/{model_name}_confusion_matrices.png')

    print(classification_report(
        total_y_true,
        total_y_pred,
        output_dict=False,
        target_names=TAGS,
        zero_division=0.0
    ))

    with open(f'../plots/{model_name}_classification_report.txt', 'w') as file:
        file.write(classification_report(
            total_y_true,
            total_y_pred,
            output_dict=False,
            target_names=TAGS,
            zero_division=0.0
        ))




