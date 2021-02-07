import torch.utils.data
import numpy as np

import steel_defect_detector as sd


class SteelDataset(torch.utils.data.Dataset):
    def __init__(self, folder, pathlist, labeled, Y_dict=None, transform=None):
        self.labeled = labeled
        self.X_path = pathlist
        self.folder = folder
        if self.labeled:
            self.Y_dict = Y_dict
        #self.X_test_path = get_data_path(test_folder)

    def __len__(self):
        return len(self.X_path)

    def __getitem__(self, idx):
        path = self.X_path[idx]
        try:
            img = sd.get_img(self.folder + '/' + path)
            img = torch.FloatTensor(img)
        except FileNotFoundError:
            print('Not found:', path)

        if self.labeled:
            labels = (np.zeros([4, 256, 1600]))
            if path in self.Y_dict:
                for classID in range(4):
                    labels[classID, :, :] = sd.rle2mask(
                        (self.Y_dict[path])[classID])
            label = torch.FloatTensor(labels).view(4, 256, 1600)
            return tuple([img, label])
        else:
            return tuple([img, path])
