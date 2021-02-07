import torch
import torch.utils.data
from sklearn.model_selection import train_test_split

import steel_defect_detector as sd


# load = 0
isGPU = torch.cuda.is_available()
print('PyTorch GPU device is available: {}'.format(isGPU))
train_folder = "../input/severstal-steel-defect-detection/train_images"
label_path = "../input/severstal-steel-defect-detection/train.csv"

print('get data path')
X_train_path = sd.get_data_path(train_folder)

print('get label')
Y_train_dict = sd.get_label(label_path)

print('split train/valid')
# X_train_path = X_train_path[:100]  # TODO: debug only
X_train_path, X_valid_path = train_test_split(X_train_path, test_size=0.1)

trainDataset = sd.SteelDataset(train_folder, X_train_path,
                               labeled=1, Y_dict=Y_train_dict)
validDataset = sd.SteelDataset(train_folder, X_valid_path,
                               labeled=1, Y_dict=Y_train_dict)

trainloader = torch.utils.data.DataLoader(trainDataset, batch_size=5,
                                          shuffle=True, num_workers=8)
validloader = torch.utils.data.DataLoader(validDataset, batch_size=5,
                                          shuffle=False, num_workers=8)

print('load model')
model = sd.Unet()
# model = torch.load(sd.MODEL_PATH)
print(model)

print('start training')
sd.Train(model, trainloader, validloader, epochs=100)
