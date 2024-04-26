import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn import Linear, Module
from torchvision.models import resnet34, ResNet34_Weights

# training machine
DEVICE = 'cuda' if torch.cuda.is_available() \
    else 'mps' if torch.backends.mps.is_available() \
    else 'cpu'

# dataset info
REAL_TRAIN_PATH = '/Users/yifan/Documents/ANU/24S1/COMP6528_Computer_Vision/wk5/Assignment_2/data/real_train'
REAL_TEST_PATH = '/Users/yifan/Documents/ANU/24S1/COMP6528_Computer_Vision/wk5/Assignment_2/data/real_test'
SKETCH_TRAIN_PATH = '/Users/yifan/Documents/ANU/24S1/COMP6528_Computer_Vision/wk5/Assignment_2/data/sketch_train'
SKETCH_TEST_PATH = '/Users/yifan/Documents/ANU/24S1/COMP6528_Computer_Vision/wk5/Assignment_2/data/sketch_test'
NUM_CLASSES = 10
CLASSES = ['backpack', 'book', 'car', 'pizza', 'sandwich', 
           'snake', 'sock', 'tiger', 'tree', 'watermelon']

# preprocessing params
IMAGE_SHAPE = (224, 224)
NORM_MEAN = (0.485, 0.456, 0.406)
NORM_STD = (0.229, 0.224, 0.225)

# train loss and acc
EPOCH_TRAIN_LOSS = []
EPOCH_TRAIN_ACC = []
EPOCH_TEST_LOSS = []
EPOCH_TEST_ACC = []


class DomainNetDataset(Dataset):
    """The customised DomainNet dataset for Task 2."""
    def __init__(self, root, transform: lambda x: x):
        # return all image info in form of [path, category]
        self.image_info = self.get_image_paths(root)
        # resize + normalise
        self.transform = transform
    def __len__(self):
        # return total number of images
        return len(self.image_info)
    def __getitem__(self, index):
        """ 
        For each data (X, y):
            X: (H, W, C) np.ndarray for each RGB image
            y: an int from range [0, ..., 9] for each class
        """
        if isinstance(index, slice):  # slicing index
            res = []
            for path, label in self.image_info[index]:
                res.append([self.transform(plt.imread(path)), label_encoding(label)])
        else:  # single index
            path, label = self.image_info[index]
            res = [self.transform(plt.imread(path)), label_encoding(label)]
        return res
    def get_image_paths(self, root):
        """For each image, retrieve its path and category info."""
        res = []
        categories = os.listdir(root)
        for category in categories:
            category_path = os.path.join(root, category)
            image_names = os.listdir(category_path)
            image_paths = [os.path.join(category_path, image_name) for image_name in image_names]
            for image_path in image_paths:
                res.append([image_path, category])
        return res
    
    
def transform(image):
    # resizing
    image = cv2.resize(image, IMAGE_SHAPE)
    # normalisation
    image = (image - NORM_MEAN) / NORM_STD
    # set to np.uint8
    image = image.astype('uint8')
    # TODO: Augmentation
    return image


def label_encoding(category):
    res = np.where(np.array(CLASSES) == category)[0][0]  # e.g. (array[8],)->8
    return res


# def mean_discrepancy_loss(mu_s, mu_t):
#     """The mean discrepancy loss implementation based on Task2.

#     Args:
#         mu_s (numpy.ndarray): The mean across feature dimensions for each source image.
#         mu_t (numpy.ndarray): The mean across feature dimensions for each target image.
    
#     Returns:
#         float: The mean discrepancy loss from mu_s and mu_t.
#     """
#     # L2 distance calculation
#     # mu_s: mean of of non-normalised logits from the batch of source data
#     # mu_t: mean of of non-normalised logits from the batch of target data
#     # l = sum((mu_s_i - mu_t_i))^2) for each entry index i
#     # mu_s and mu_t should share the same dimension
#     if type(mu_s) == torch.Tensor:
#         mu_s = mu_s.numpy()
#     if type(mu_t) == torch.Tensor:
#         mu_t = mu_t.numpy()
#     return 


class MaximumMeanDiscrepancyLoss(Module):
    def __init__(self):
        super(MaximumMeanDiscrepancyLoss, self).__init__()

    def forward(self, outputs, targets):
        # mean of source logits
        mu_s = torch.as_tensor(outputs, dtype=torch.float32).mean(axis=0)
        # mean of target logits
        mu_t = torch.as_tensor(targets, dtype=torch.float32).mean(axis=0)
        # return the L2 norm of the difference between mu_s and mu_t
        return torch.norm(mu_s - mu_t)


def plot_metric(train_loss, train_acc, test_loss, test_acc, title="ResNet-34 on DomainNet"):
    fig, ax = plt.subplots(2, 2, figsize=(10, 9))
    ax[0, 0].plot(train_loss)
    ax[0, 0].set_xlabel("epoch")
    ax[0, 0].set_ylabel("loss")
    ax[0, 0].set_title("training set")
    ax[1, 0].plot(test_loss)
    ax[1, 0].set_xlabel("epoch")
    ax[1, 0].set_ylabel("loss")
    ax[1, 0].set_title("test set")
    ax[0, 1].plot(train_acc)
    ax[0, 1].set_xlabel("epoch")
    ax[0, 1].set_ylabel("accuracy")
    ax[0, 1].set_title("training set")
    ax[1, 1].plot(test_acc)
    ax[1, 1].set_xlabel("epoch")
    ax[1, 1].set_ylabel("accuracy")
    ax[1, 1].set_title("test set")
    fig.suptitle(title)
    plt.show()
    

def ResNet34DomainNet(weights=ResNet34_Weights.DEFAULT):
    # load the latest ResNet-34 pretrained model
    model = resnet34(weights=weights)
    # modify the last FC layer to fit the 10 classes
    model.fc = Linear(model.fc.in_features, NUM_CLASSES)
    return model


def get_dataloader(path, batch_size, transform=transform):
    dataset = DomainNetDataset(path, transform)
    dataloader = DataLoader(dataset, batch_size, shuffle=True)
    return dataloader


def train_loop(train_loader, model, loss_fn, optimiser):
    model.train()
    model.to(DEVICE)
    BATCH_TRAIN_LOSS = []
    BATCH_TRAIN_ACC = []
    # training set size
    size = len(train_loader.dataset)
    for batch, (X, y) in enumerate(train_loader):
        # X shape: (B:64, H:224, W:224, C:3) -> (B:64, C:3, H:224, W:224)
        # y: an int from [0, ..., 9]
        X, y = torch.from_numpy(np.transpose(X.numpy().astype('float32'), (0, 3, 1, 2))).to(DEVICE), y.to(DEVICE)
        # forward propagation
        pred = model(X)
        # cross-entropy loss
        loss = loss_fn(pred, y)
        # backward propagation and calculate gradients
        loss.backward()
        # update params
        optimiser.step()
        # reset gradients
        optimiser.zero_grad()
        # cross-entropy loss from this batch
        loss = loss.item()
        BATCH_TRAIN_LOSS.append(loss)
        # accuracy from this batch
        acc =  (pred.argmax(1) == y).type(torch.float).sum().item() / len(X)
        BATCH_TRAIN_ACC.append(acc)
        # print results
        if batch % 50 == 0:
            current = batch * train_loader.batch_size + len(X)
            print(f"train loss: {loss:>7f},  acc: {acc:>7f}  [{current:>5d} / {size:>5d}]")
    avg_train_loss = np.mean(BATCH_TRAIN_LOSS)
    avg_train_acc = np.mean(BATCH_TRAIN_ACC)
    print(f"avg train loss: {avg_train_loss:>7f},  avg train acc: {avg_train_acc:>7f}")
    EPOCH_TRAIN_LOSS.append(avg_train_loss)
    EPOCH_TRAIN_ACC.append(avg_train_acc)


def test_loop(test_loader, model, loss_fn):
    model.eval()
    model.to(DEVICE)
    BATCH_TEST_LOSS = []
    BATCH_TEST_ACC = []
    # test set size
    size = len(test_loader.dataset)
    with torch.no_grad():
        for batch, (X, y) in enumerate(test_loader):
            # X shape: (B:64, H:224, W:224, C:3) -> (B:64, C:3, H:224, W:224)
            # y: an int from [0, ..., 9]
            X, y = torch.from_numpy(np.transpose(X.numpy().astype('float32'), (0, 3, 1, 2))).to(DEVICE), y.to(DEVICE)
            pred = model(X)
            # cross-entropy loss from this batch
            loss = loss_fn(pred, y)
            loss = loss.item()
            BATCH_TEST_LOSS.append(loss)
            # accuracy from this batch
            acc =  (pred.argmax(1) == y).type(torch.float).sum().item() / len(X)
            BATCH_TEST_ACC.append(acc)
            # print results
            if batch % 50 == 0:
                current = batch * test_loader.batch_size + len(X)
                print(f"test loss: {loss:>7f}, acc: {acc:>7f}    [{current:>5d} / {size:>5d}]")
    avg_test_loss = np.mean(BATCH_TEST_LOSS)
    avg_test_acc = np.mean(BATCH_TEST_ACC)
    print(f"avg test loss: {avg_test_loss:>7f},  avg test acc: {avg_test_acc:>7f}")
    EPOCH_TEST_LOSS.append(avg_test_loss)
    EPOCH_TEST_ACC.append(avg_test_acc)


if __name__ == '__main__':
    pass