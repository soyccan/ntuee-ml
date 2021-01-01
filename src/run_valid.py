import torch.nn as nn
import torch.utils.data

import steel_defect_detector as sd


bceloss = nn.BCEWithLogitsLoss()
sigmoid = nn.Sigmoid()

train_folder = "../input/severstal-steel-defect-detection/train_images"
label_path = "../input/severstal-steel-defect-detection/train.csv"

X_train_path = sd.get_data_path(train_folder)
Y_train_dict = sd.get_label(label_path)

validDataset = sd.SteelDataset(train_folder, X_train_path,
                               labeled=1, Y_dict=Y_train_dict)

V = len(X_train_path)
validloader = torch.utils.data.DataLoader(validDataset, batch_size=4,
                                          shuffle=False, num_workers=4)
model = torch.load(sd.MODEL_PATH)
sd.valid_test(model, validloader, V, 100)
