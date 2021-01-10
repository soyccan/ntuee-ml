import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import joblib
import torch
import torch.utils.data
from torch.optim import *
from torch.nn import *
import torch.nn.functional as F

from rnn.util import *


import torch

class FastTensorDataLoader:
    """
    https://github.com/hcarlens/pytorch-tabular/blob/master/fast_tensor_data_loader.py
    A DataLoader-like object for a set of tensors that can be much faster than
    TensorDataset + DataLoader because dataloader grabs individual indices of
    the dataset and calls cat (slow).
    Source: https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/6
    """
    def __init__(self, dataset, batch_size=32, shuffle=False, **kwargs):
        """
        Initialize a FastTensorDataLoader.
        :param *tensors: tensors to store. Must have the same length @ dim 0.
        :param batch_size: batch size to load.
        :param shuffle: if True, shuffle the data *in-place* whenever an
            iterator is created out of this object.
        :returns: A FastTensorDataLoader.
        """
        tensors = [dataset.X, dataset.y]
        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
        self.tensors = tensors

        self.dataset_len = self.tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Calculate # batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches

    def __iter__(self):
        if self.shuffle:
            r = torch.randperm(self.dataset_len)
            self.tensors = [torch.from_numpy(t[r]) for t in self.tensors]
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration
        batch = (
            torch.from_numpy(self.tensors[0][self.i:self.i+self.batch_size].todense()),
            torch.from_numpy(self.tensors[1][self.i:self.i+self.batch_size]))
        self.i += self.batch_size
        return batch

    def __len__(self):
        return self.n_batches

class SparseDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return (torch.from_numpy(self.X[idx].todense()),
                torch.from_numpy(self.y[idx]))

class BowModel(Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout=0.5):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        self.classifier = Sequential()
        self.classifier.add_module('dropout', Dropout(dropout))
        self.classifier.add_module('input', Linear(input_dim, hidden_dim[0]))
        for i in range(len(hidden_dim)-1):
            self.classifier.add_module(
                'hidden_'+str(i), Linear(hidden_dim[i], hidden_dim[i+1]))
        self.classifier.add_module('output', Linear(hidden_dim[-1], 1))
        self.classifier.add_module('output_activation', Sigmoid())

    def forward(self, inputs):
        x = self.classifier(inputs)
        return x


def evaluation(outputs, labels):
    # outputs => probability (float)
    # labels => labels
    outputs[outputs >=0.5] = 1 # 大於等於 0.5 為有惡意
    outputs[outputs < 0.5] = 0 # 小於 0.5 為無惡意
    correct = torch.sum(torch.eq(outputs, labels)).item()
    return correct


def training(batch_size, n_epoch, lr, model_dir,
             train_loader, valid_loader, model, device):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('\nstart training, parameter total:{}, trainable:{}\n'.format(
        total, trainable))
    model.train()
    criterion = BCELoss()
    t_batch = len(train_loader)
    v_batch = len(valid_loader)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-3)
    total_loss, total_acc, best_acc = 0, 0, 0
    for epoch in range(n_epoch):
        # training
        total_loss, total_acc = 0, 0
        # start loading the first batch
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device, dtype=torch.float, non_blocking=True)
            labels = labels.to(device, dtype=torch.float, non_blocking=True)
            optimizer.zero_grad()
            outputs = model(inputs)
            # print('outputs',outputs)
            outputs = outputs.squeeze() # 去掉最外面的 dimension，好讓 outputs 可以餵進 criterion()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            correct = evaluation(outputs, labels)
            total_acc += correct / batch_size
            total_loss += loss.item()
            print('[ Epoch{}: {}/{} ] loss:{:.3f} acc:{:.3f} '.format(
                epoch+1, i+1, t_batch, loss.item(), correct*100/batch_size), end='\r')
        print('\nTrain | Loss:{:.5f} Acc: {:.3f}'.format(
            total_loss/t_batch, total_acc/t_batch*100))

        # validation
        model.eval()
        with torch.no_grad():
            total_loss, total_acc = 0, 0
            for i, (inputs, labels) in enumerate(valid_loader):
                inputs = inputs.to(device, dtype=torch.float)
                labels = labels.to(device, dtype=torch.float)
                outputs = model(inputs)
                outputs = outputs.squeeze() # 去掉最外面的 dimension，好讓 outputs 可以餵進 criterion()
                loss = criterion(outputs, labels)
                correct = evaluation(outputs, labels)
                total_acc += (correct / batch_size)
                total_loss += loss.item()

            print("Valid | Loss:{:.5f} Acc: {:.3f} ".format(
                total_loss/v_batch, total_acc/v_batch*100))
            if total_acc > best_acc:
                best_acc = total_acc
                #torch.save(model, "{}/val_acc_{:.3f}.model".format(model_dir,total_acc/v_batch*100))
                torch.save(model, "{}/ckpt.model".format(model_dir))
                print('saving model with acc {:.3f}'.format(total_acc/v_batch*100))
        print('-----------------------------------------------')
        model.train()

def bag_of_words(corpus):
    # if os.path.exists('../work/bow.pkl'):
    #     return joblib.load('../work/bow.pkl')
    vectorizer = CountVectorizer()
    word_vectors = vectorizer.fit_transform(corpus)
    joblib.dump(vectorizer, 'bow.pkl')
    print('Features: \n', vectorizer.get_feature_names())
    return word_vectors  # sparse matrix

def sparse2tensor_long(X):
    X = X.tocoo()
    return torch.sparse.LongTensor(
        torch.LongTensor([X.row.tolist(), X.col.tolist()]),
        torch.LongTensor(X.data.astype(np.int32)))

def sparse2tensor_float(X):
    X = X.tocoo()
    return torch.sparse.FloatTensor(
        torch.LongTensor([X.row.tolist(), X.col.tolist()]),
        torch.FloatTensor(X.data.astype(np.float32)))

def main():
    # set GPU to use
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    print("loading data ...")
    X, y = load_training_data('../work/training_label.txt')
    X = bag_of_words([' '.join(ln) for ln in X])
    y = np.array(y, dtype='float32')
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    # X_train = sparse2tensor_float(X_train)
    # X_val = sparse2tensor_float(X_val)
    # y_train = torch.FloatTensor(y_train)
    # y_val = torch.FloatTensor(y_val)

    train_dataset = SparseDataset(X=X_train, y=y_train)
    val_dataset = SparseDataset(X=X_val, y=y_val)

    batch_size = 256
    train_loader = FastTensorDataLoader(dataset=train_dataset,
                                        batch_size=batch_size,
                                        shuffle=False,
                                        num_workers=8,
                                        pin_memory=True)
    val_loader = FastTensorDataLoader(dataset=val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=8)

    model = BowModel(input_dim=X_train.shape[1],
                     hidden_dim=(128, 128, 128, 128), num_layers=0, dropout=0.5)
    model = model.to(device)

    training(batch_size=batch_size, n_epoch=100, lr=1e-3,
             model_dir='.',
             train_loader=train_loader, valid_loader=val_loader,
             model=model, device='cuda')


main()
