import torch
import numpy as np
import torch.nn.functional as F


def fit(train_loader, val_loader, model, loss_fn, optimizer, n_epochs, device, log_interval, metrics=None):
    """
    Loaders, model, loss function and metrics should work together for a given task,
    i.e. the model should be able to process data output of loaders,
    loss function should process target output of loaders and outputs from the model
    Examples:
    Classification: batch loader, classification model, NLL loss, accuracy metric.
    Siamese network: Siamese loader, siamese model, contrastive loss.
    Online triplet learning: batch loader, embedding model, online triplet loss.
    """
    for epoch in range(n_epochs):
        # training step: compute training average loss and metrics
        train_loss, train_metrics = train_epoch(train_loader, model, loss_fn, optimizer, device, log_interval, metrics)
        # train_loss /= len(train_loader)  # TODO: check what loss is returned
        message = 'Epoch: {}/{}. Training average loss: {:.4f}'.format(epoch + 1, n_epochs, train_loss)
        for metric in train_metrics:
            message += '\t{}: {}'.format(metric.name(), metric.value())
        # validation step: compute validation average loss and metrics
        val_loss, val_metrics = test_epoch(val_loader, model, loss_fn, device, metrics)
        val_loss /= len(val_loader)  # TODO: check what loss is returned
        message += '\nEpoch: {}/{}. Validation average loss: {:.4f}'.format(epoch + 1, n_epochs, val_loss)
        for metric in val_metrics:
            message += '\t{}: {}'.format(metric.name(), metric.value())
        print(message)
        # save trained weights


def train_epoch(train_loader, model, loss_fn, optimizer, device, log_interval, metrics=None):
    # reset metrics values before starting new computation
    if metrics:
        for metric in metrics:
            metric.reset()
    # set model in training mode
    model.train()
    total_loss = 0
    losses = []

    for batch_idx, (data, target) in enumerate(train_loader):
        # data and target must be a tuples
        target = (target.to(device),) if len(target) > 0 else None
        if type(data) not in (tuple, list):
            data = (data,)
        data = tuple(d.to(device) for d in data)

        optimizer.zero_grad()
        outputs = model(*data)

        loss_inputs = outputs
        if target is not None:
            loss_inputs += target

        loss_outputs = loss_fn(*loss_inputs)
        loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
        losses.append(loss.item())
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

        for metric in metrics:
            # target = target[0] if type(target) in (tuple, list) else target
            metric(outputs, target)

        if batch_idx % log_interval == 0:
            message = 'Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                batch_idx * len(data[0]), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), np.mean(losses))
            for metric in metrics:
                message += '\t{}: {}'.format(metric.name(), metric.value())
            # print(message)
            losses = []

    total_loss /= batch_idx + 1  # TODO: check loss
    return total_loss, metrics


def test_epoch(val_loader, model, loss_fn, device, metrics=None):
    model.eval()
    with torch.no_grad():
        for metric in metrics:
            metric.reset()
        val_loss = 0

        for batch_idx, (data, target) in enumerate(val_loader):
            target = (target.to(device),) if len(target) > 0 else None
            if type(data) not in (tuple, list):
                data = (data,)
            data = tuple(d.to(device) for d in data)

            outputs = model(*data)
            if type(outputs) not in (tuple, list):
                outputs = (outputs,)
            loss_inputs = outputs

            if target:
                loss_inputs += target

            loss_outputs = loss_fn(*loss_inputs)
            loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
            val_loss += loss.item()

            for metric in metrics:
                # target = target[0] if type(target) in (tuple, list) else target
                metric(outputs, target)

    return val_loss, metrics


