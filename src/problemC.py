"""Problem c (作圖)
===
將 train data 的降維結果 (embedding) 與他們對應的 label 畫出來。
"""
import numpy as np

trainX = np.load('trainX.npy')
trainY = np.load('trainY.npy')

model.load_state_dict(torch.load('./checkpoints/last_checkpoint.pth'))
model.eval()
latents = inference(trainX, model)
pred_from_latent, emb_from_latent = predict(latents)
acc_latent = cal_acc(trainY, pred_from_latent)
print('The clustering accuracy is:', acc_latent)
print('The clustering result:')
plot_scatter(emb_from_latent, trainY, savefig='baseline.png')
