import torch.utils.data
import steel_defect_detector as sd

test_folder = "../input/severstal-steel-defect-detection/test_images"
X_test_path = sd.get_data_path(test_folder)
# X_test_path.sort()

testDataset = sd.SteelDataset(test_folder, X_test_path, labeled=0)
testloader = torch.utils.data.DataLoader(testDataset, batch_size=5,
                                         shuffle=False, num_workers=8)

print("X_test size:{}".format(len(X_test_path)))
print(X_test_path)

model = torch.load(sd.MODEL_PATH)
sd.Test(model, testloader, outpath=sd.SUBMISSION_PATH)
