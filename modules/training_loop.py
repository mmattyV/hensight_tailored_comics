from constants import *

def train_model(model, criterion, optimizer, lr_scheduler, dsets, dset_sizes, dloads, num_epochs=10, device=None):
    since = time.time()

    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    model.to(device)

    best_model = copy.deepcopy(model)
    best_ham = torch.inf

    accuracies = {'train': [], 'val': []}
    losses = {'train': [], 'val': []}
    hams = {'train': [], 'val': []}

    epoch_bar = tqdm(total=num_epochs, position=3)

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
            running_acc = 0.0
            running_ham = 0.0

            print(f'\nIn {phase} phase:')
            batch_bar = tqdm(total=dsets[phase].__len__(), position=2)
            
            for inputs, labels in dloads[phase]:
                inputs = inputs.float().to(device)  # Move inputs to device and ensure data type is float.
                labels = labels.long().to(device)  # Move labels to device and ensure data type is long.

                optimizer.zero_grad()

                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)  # Get model outputs.
                    loss = criterion(outputs, labels.float())  # Calculate loss.

                    # Backward pass and optimize only if in training phase.
                    if phase == 'train':
                        loss.backward()  # Compute gradients.
                        optimizer.step()  # Update parameters.

                batch_bar.update(BATCH_SIZE)

                y_true = labels.to('cpu').numpy()
                y_pred = outputs.to('cpu').detach().numpy()

                y_pred = (y_pred > 0).astype(int)

                running_loss += loss.item() * inputs.size(0)
                running_ham += hamming_loss(y_true, y_pred)
                running_acc += accuracy_score(y_true, y_pred)

            batch_bar.close()

            # Calculate average loss and accuracy for the current phase.
            epoch_loss = running_loss / dset_sizes[phase]
            epoch_ham = running_ham / dset_sizes[phase]
            epoch_acc = running_acc / dset_sizes[phase]
            hams[phase].append(epoch_ham)  # Store hamming loss.
            losses[phase].append(epoch_loss)  # Store loss.
            accuracies[phase].append(epoch_acc) # Store accuracy

            print('{} Loss: {:.4f} Hamming loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_ham, epoch_acc))

            # Check if the current model is the best model; if so, update the best model.
            if phase == 'val' and epoch_ham < best_ham:
                best_ham = epoch_ham
                best_model = copy.deepcopy(model)
                print('New best hamming loss = {:.4f}'.format(best_ham))

        epoch_bar.update(1)

    epoch_bar.close()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best ham loss: {:4f}'.format(best_ham))
    print('returning and looping back')

    return best_model, accuracies, losses, hams

def exp_lr_scheduler(optimizer, epoch, init_lr=BASE_LR, lr_decay_epoch=EPOCH_DECAY):
    """Decay learning rate by a factor of DECAY_WEIGHT every lr_decay_epoch epochs."""
    lr = init_lr * (DECAY_WEIGHT**(epoch // lr_decay_epoch))

    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer