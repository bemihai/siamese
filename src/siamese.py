""" Siamese network on MNIST data """

import torch
from torch import optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.backends import cudnn

from datasets import BalancedPairs
from trainer import train, test, oneshot
from networks import FeatureExtractor, SiameseNetBin

cudnn.benchmark = True
do_learn = True
batch_size = 256
lr = 0.01
num_epochs = 10
weight_decay = 0.0001

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
feature_extractor = FeatureExtractor()
model = SiameseNetBin(feature_extractor).to(device)

def main():

    if do_learn:  # training mode
        # load data
        training_data = BalancedPairs('../data', n_pairs=2000, train=True, transform=trans, save_pairs=True)
        testing_data = BalancedPairs('../data', n_pairs=500, train=False, transform=trans, save_pairs=True)

        train_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(testing_data, batch_size=batch_size, shuffle=False)

        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        for epoch in range(num_epochs):
            train(model, device, train_loader, batch_size, epoch, optimizer)
            test(model, device, test_loader)

        # save trained model
        torch.save(model.state_dict(), '../models/siamese.pt')

    else:  # prediction mode
        testing_data = BalancedPairs('../data', n_pairs=100, train=False, transform=trans, save_pairs=True)
        prediction_loader = torch.utils.data.DataLoader(testing_data, batch_size=1, shuffle=True)
        model.load_state_dict(torch.load('../models/siamese.pt'))
        data = []
        data.extend(next(iter(prediction_loader))[0][:3:2])
        same = oneshot(model, device, data)
        if same > 0:
            print('These two images are of the same number')
        else:
            print('These two images are not of the same number')


if __name__ == '__main__':
    main()



