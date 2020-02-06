import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
from tqdm import tqdm, trange

from delve import CheckLayerSat

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

batch_size = 128

train_set = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=batch_size, shuffle=True, num_workers=2)

test_set = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(
    test_set, batch_size=batch_size, shuffle=False, num_workers=2)


class Net(nn.Module):
    def __init__(self, h2):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, h2)
        self.fc3 = nn.Linear(h2, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(1)
    if not os.path.exists('convNet'):
        os.mkdir('convNet')

    epochs = 5

    for h2 in [8, 32, 128]:  # compare various hidden layer sizes
        net = resnet18(pretrained=False, num_classes=10)#Net(h2=h2)  # instantiate network with hidden layer size `h2`

        net.to(device)
        logging_dir = 'convNet/simpson_h2-{}'.format(h2)
        stats = CheckLayerSat(savefile=logging_dir, save_to='plot', modules=net, include_conv=True, stats=['lsat'], max_samples=1024,
                              verbose=True, writer_args={'figsize': [30, 30], 'fontsize': 32}, conv_method='mean', device='cpu')

        #net = nn.DataParallel(net, device_ids=['cuda:0', 'cuda:1'])
        print(net)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

        #stats.write( "CIFAR10 ConvNet - Changing fc2 - size {}".format(h2))  # optional

        for epoch in range(epochs):
            running_loss = 0.0
            step = 0
            loader = tqdm(
                train_loader, leave=True,
                position=0)  # track step progress and loss - optional
            for i, data in enumerate(loader):
                step = epoch * len(loader) + i
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()

                optimizer.step()

                running_loss += loss.data
                if i % 2000 == 1999:  # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1,
                                                    running_loss / 2000))
                    running_loss = 0.0

                # update the training progress display
                loader.set_description(desc='[%d/%d, %5d] loss: %.3f' %
                                       (epoch + 1, epochs, i + 1, loss.data))
                # display layer saturation levels

            stats.add_scalar('epoch', epoch)  # optional
            stats.add_scalar('loss', running_loss.cpu().numpy())  # optional
            stats.add_saturations()

        loader.write('\n')
        loader.close()
        stats.close()