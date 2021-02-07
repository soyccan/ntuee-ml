import torch.optim
import torch.utils.data

import steel_defect_detector as sd


def Train(model, trainloader, validloader, epochs):
    best = 100
    decay = 1e-4
    LR = 0.001
    # optim = torch.optim.Adam( add_weight_decay(model,decay) , lr= LR)
    optim = torch.optim.Adam(model.parameters(), lr=LR)
    model.cuda()
    t_batch = len(trainloader.dataset) // trainloader.batch_size
    print('Start training lr={} decay={} trainable={}'.format(
          LR, decay,
          sum(x.numel() for x in model.parameters())))
    history = open('history.csv', 'w')
    history.write('train_loss,val_loss,val_dice\n')
    for epoch in range(int(epochs)):
        model.train()
        for i, (batch_x, batch_y) in enumerate(trainloader):
            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()

            optim.zero_grad()
            y_hat = model(batch_x)
            loss = sd.weighted_bceloss(y_hat, batch_y)
            # loss = bce_dice_loss(y_hat,batch_y)
            loss.backward()
            optim.step()

            print('Epoch {}: {}/{} train_loss: {:.3f}'.format(
                    epoch, i, t_batch, loss.item()),
                  end='\r')

        val_loss, val_dice = sd.valid_test(model, validloader, epoch)
        print("Epoch {} valid_loss: {:.3f}, valid_dice: {:.3f}".format(
                epoch, val_loss, val_dice))
        history.write('{},{},{}\n'.format(loss, val_loss, val_dice))
        if loss < best:
            best = loss
            print("Saving model, valid_loss: {:.3f}, valid_dice: {:.3f}".format(
                    val_loss, val_dice))
            torch.save(model, 'best.pt')
    history.close()

def Train_plus(model, trainloader):
    epochs = 3
    best = 100
    decay = 1e-4
    LR = 0.001
    # optim = torch.optim.Adam( add_weight_decay(model,decay) , lr= LR)
    optim = torch.optim.Adam(model.parameters(), lr=LR)
    print('train-------------lr:', LR, 'decay:', decay)
    model.cuda()
    for epoch in range(int(epochs)):
        model.train()
        for i, (batch_x, batch_y) in enumerate(trainloader):

            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()

            optim.zero_grad()
            y_hat = model(batch_x)
            loss = sd.weighted_bceloss(y_hat, batch_y)
            # loss = bce_dice_loss(y_hat,batch_y)
            loss.backward()
            optim.step()

            if i % 100 == 99:
                loss = sd.valid_test(epoch)
                if loss < best:
                    best = loss
                    print("saving model with loss: {:.3f}".format(loss))
                    torch.save(model, 'model2.pth')
