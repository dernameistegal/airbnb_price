import torch
import fastprogress
import numpy as np
import time
import os
import copy

scaler = torch.cuda.amp.GradScaler()


############################################################################
##########################run training with embeddings
############################################################################


def train(dataloader, optimizer, model, loss_fn, device, ntrain, master_bar):
    """Run one training epoch."""

    epoch_loss = []
    i = 0

    for pic_embdg, description_embdg, reviews_embdg, cont_features, cat_features, label in fastprogress.progress_bar(
            dataloader, parent=master_bar):
        pic_embdg, description_embdg, reviews_embdg, cont_features, cat_features, label = \
            pic_embdg.to(device), description_embdg.to(device), reviews_embdg.to(device), \
            cont_features.to(device), cat_features.to(device).int(), label.to(device)

        optimizer.zero_grad()
        model.train()

        # Forward pass
        with torch.cuda.amp.autocast():
            labels_pred = model(pic_embdg, description_embdg, reviews_embdg, cont_features, cat_features)
            labels_pred = torch.squeeze(labels_pred)
            loss = loss_fn(labels_pred, torch.squeeze(label))

        # Backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # For plotting the train loss, save it for each sample
        epoch_loss.append(loss.item())
        i = i + 1
        if i > ntrain:
            break

    # Return the mean loss and the accuracy of this epoch
    return np.mean(epoch_loss), np.sqrt(np.mean(epoch_loss))


def validate(dataloader, model, loss_fn, device, nval, master_bar):
    epoch_loss = []

    model.eval()
    i = 0
    with torch.no_grad():
        for pic_embdg, description_embdg, reviews_embdg, cont_features, cat_features, label in fastprogress.progress_bar(
                dataloader, parent=master_bar):
            pic_embdg, description_embdg, reviews_embdg, cont_features, cat_features, label = \
                pic_embdg.to(device), description_embdg.to(device), reviews_embdg.to(device), \
                cont_features.to(device), cat_features.to(device).int(), label.to(device)
            # make a prediction on validation set
            labels_pred = model(pic_embdg, description_embdg, reviews_embdg, cont_features, cat_features)
            labels_pred = torch.squeeze(labels_pred)

            # Compute loss
            loss = loss_fn(labels_pred, torch.squeeze(label))

            # For plotting the train loss, save it for each sample
            epoch_loss.append(loss.item())
            i = i + 1
            if i > nval:
                break

    return np.mean(epoch_loss), np.sqrt(np.mean(epoch_loss))


def run_training(model, optimizer, loss_function, device, num_epochs,
                 train_dataloader, val_dataloader, ntrain=150, nval=50, verbose=False, scheduler=None,
                 savefolder="test"):
    # make path to save performance measures and state dict
    savepath = "/content/drive/MyDrive/Colab/airbnb/data/model_performance/" + savefolder
    os.mkdir(savepath)

    start_time = time.time()
    master_bar = fastprogress.master_bar(range(num_epochs))
    train_losses, val_losses, train_rmse, val_rmse = [], [], [], []
    for epoch in master_bar:
        # Train the model
        epoch_train_loss, epoch_train_loss_sqrt = train(train_dataloader, optimizer, model,
                                                        loss_function, device, ntrain, master_bar)
        # Validate the model
        epoch_val_loss, epoch_val_loss_sqrt = validate(val_dataloader, model, loss_function, device, nval, master_bar)

        # Save loss and acc for plotting and add increase of val_acc
        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        train_rmse.append(epoch_train_loss_sqrt)
        val_rmse.append(epoch_val_loss_sqrt)

        if val_rmse[-1] <= np.min(val_rmse):
            checkpoint = model.state_dict()
            wholemodel = copy.deepcopy(model)
            print("saving model...")

        if scheduler:
            scheduler.step()
        if verbose:
            master_bar.write(
                f'Epoch: {epoch}, Train loss: {epoch_train_loss:.2f}, val loss: {epoch_val_loss:.2f}, train rmse: {epoch_train_loss_sqrt:.3f}, val rmse {epoch_val_loss_sqrt:.3f}')

    np.save(savepath + "/trainloss", train_rmse)
    np.save(savepath + "/valloss", val_rmse)
    torch.save(checkpoint, savepath + "/checkpoint.pt")
    torch.save(wholemodel, savepath + "/wholemodel.pt")

    time_elapsed = np.round(time.time() - start_time, 0).astype(int)
    print(f'Finished training after {time_elapsed} seconds.')
    return train_losses, val_losses, train_rmse, val_rmse
