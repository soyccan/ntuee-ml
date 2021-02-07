import matplotlib.pyplot as plt

# model = torch.load("../input/model/model2.3.pth")
fig, axs = plt.subplots(5, figsize=(12, 12))

sample_img = get_img(
    '../input/severstal-steel-defect-detection/train_images/0025bde0c.jpg')
# predlist = sigmoid(model(torch.from_numpy(sample_img.astype('float32')).view(-1,1,256,1600).cuda()))

label_path = "../input/severstal-steel-defect-detection/train.csv"
path = "0025bde0c.jpg"
Y_dict = get_label(label_path)

print(Y_dict[path])
labels = (np.zeros([4, 256, 1600]))
for classID in range(4):
    rle = (Y_dict[path])[classID]
    print(rle)
    labels[classID, :, :] = rle2mask((Y_dict[path])[classID])

predlist = labels

# print(predlist)
for n in range(len(predlist)):
    print(predlist.shape)
    pred = predlist[n]
    for ClassId in range(0, 4):
        print(labels[ClassId].sum())
        # p = (pred[ClassId].flatten(1) > 0.5).float()
        # img = np.array(p.cpu()).reshape(256,1600)
        # encoded_value = mask2rle(p.cpu())
        # if len(encoded_value) > 0:
        # img = rle2mask(encoded_value)
        # print(img)
        axs[ClassId + 1].imshow(pred)
        axs[ClassId + 1].axis('off')
axs[0].imshow(sample_img)
axs[0].axis('off')
