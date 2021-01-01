import steel_defect_detector as sd
import torch


def valid_test(model, validloader, epoch):
    model.eval()
    batch_size = 10
    Totaldice = 0
    Totalloss = 0
    sigmoid = torch.nn.Sigmoid()
    V = len(validloader.dataset)
    t_batch = V // validloader.batch_size
    for i, (batch_x, batch_y) in enumerate(validloader):
        batch_x = batch_x.cuda()
        batch_y = batch_y.cuda()
        pV = model(batch_x)
        loss = sd.weighted_bceloss(pV, batch_y).item()
        # loss = bce_dice_loss(pV,batch_y).item()
        dice = sd.dice_channel_torch(sigmoid(pV), batch_y, [0.5, 0.5, 0.5, 0.5])
        Totalloss += loss * len(batch_x)
        Totaldice += dice * len(batch_x)

        print('Epoch{}: {}/{} loss: {:.3f} dice: {:.3f}'.format(
                epoch, i, t_batch, loss, dice),
              end='\r')

    meandice = Totaldice / V
    meanloss = Totalloss / V
    # print(pV)
    # print('Epoch: {} Valid dice: {:.3f} loss: {:.3f}'.format(
    #     epoch, meandice, meanloss))  # time.time()-start_time
    model.train()
    return meanloss, meandice
