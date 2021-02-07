import steel_defect_detector as sd
from torch import nn


def Test(model, testloader, outpath):
    model.eval()
    sigmoid = nn.Sigmoid()
    threshold = [0.5, 0.5, 0.5, 0.5]
    with open(outpath, "w") as f:
        f.write("ImageId_ClassId,EncodedPixels\n")
        for i, (img, path) in enumerate(testloader):
            print(i, path)
            predlist = sigmoid(model(img.cuda()))
            for n in range(predlist.shape[0]):
                pred = predlist[n]
                for ClassId in range(0, 4):
                    p = (pred[ClassId] > threshold[ClassId])
                    encoded_value = sd.mask2rle(p.cpu())
                    # if len(encoded_value) > 0:
                    f.write("{}_{},{}\n".format(
                        path[n], ClassId + 1, encoded_value))
