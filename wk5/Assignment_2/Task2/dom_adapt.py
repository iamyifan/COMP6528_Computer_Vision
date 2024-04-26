from utils import *
from torch.optim import SGD, Adam
from torch.nn import CrossEntropyLoss


if __name__ == '__main__':
    EPOCHS = 50
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-5
    MODEL = ResNet34DomainNet()
    # OPTIMISER = SGD(MODEL.parameters(), LEARNING_RATE)
    OPTIMISER = Adam(MODEL.parameters(), LEARNING_RATE)
    CLS_LOSS_FN = CrossEntropyLoss()
    MMD_LOSS_FN = MaximumMeanDiscrepancyLoss()
    MMD_LOSS_WEIGHT = 0.01
    
    real_train_loader = get_dataloader(REAL_TRAIN_PATH, BATCH_SIZE)
    real_test_loader = get_dataloader(REAL_TEST_PATH, BATCH_SIZE)
    sketch_train_loader = get_dataloader(SKETCH_TRAIN_PATH, BATCH_SIZE)
    sketch_test_loader = get_dataloader(SKETCH_TEST_PATH, BATCH_SIZE)
    
    # train loss and acc
    DA_EPOCH_TRAIN_LOSS = []
    DA_EPOCH_TRAIN_ACC = []
    DA_EPOCH_TEST_LOSS = []
    DA_EPOCH_TEST_ACC = []
    
    for epoch in range(EPOCHS):
        """ train """
        MODEL.train()
        MODEL.to(DEVICE)
        DA_BATCH_CLS_LOSS = []
        DA_BATCH_MMD_LOSS = []
        DA_BATCH_JON_LOSS = []
        DA_BATCH_TEST_LOSS = []
        DA_BATCH_TRAIN_ACC = []
        DA_BATCH_TEST_ACC = []
        print(f"--------------- epoch {epoch+1} ---------------")
        for batch, ((X_real, y_real), (X_sketch, y_sketch)) in enumerate(zip(real_train_loader, sketch_train_loader)):
            # import data from both datasets
            X_real = torch.from_numpy(np.transpose(X_real.numpy().astype('float32'), (0, 3, 1, 2))).to(DEVICE)
            y_real = y_real.to(DEVICE)
            X_sketch = torch.from_numpy(np.transpose(X_sketch.numpy().astype('float32'), (0, 3, 1, 2))).to(DEVICE)
            # logits to be calculated in MMD loss function (BATCH_SIZE, NUM_CLASSES)
            pred_real = MODEL(X_real)
            pred_sketch = MODEL(X_sketch)
            # MMD loss: L2 dist between mean of pred_real and mean of pred_sketch
            mmd_loss = MMD_LOSS_FN(pred_real, pred_sketch)
            # classification loss: only source logits, pred_real and y_real
            cls_loss = CLS_LOSS_FN(pred_real, y_real)
            # joint loss
            joint_loss = cls_loss + MMD_LOSS_WEIGHT * mmd_loss
            # bp
            joint_loss.backward()
            # update parameters
            OPTIMISER.step()
            OPTIMISER.zero_grad()
            # loss
            loss_mdd = mmd_loss.item()
            DA_BATCH_MMD_LOSS.append(loss_mdd)
            loss_cls = cls_loss.item()
            DA_BATCH_CLS_LOSS.append(loss_cls)
            loss_joint = joint_loss.item()
            DA_BATCH_JON_LOSS.append(loss_joint)
            # acc for real datasets
            acc = (pred_real.argmax(1) == y_real).type(torch.float).sum().item() / len(X_real)
            DA_BATCH_TRAIN_ACC.append(acc)
            # print results
            if batch % 50 == 0:
                current_real = batch * real_train_loader.batch_size + len(X_real)
                current_sketch = batch * sketch_train_loader.batch_size + len(X_sketch)
                print(f"train loss {loss_joint:>7f} ({loss_cls:>3f}+{MMD_LOSS_WEIGHT}*{loss_mdd:>3f}), acc: {acc:>7f}")
        avg_train_loss = np.mean(DA_BATCH_JON_LOSS)
        avg_train_acc = np.mean(DA_BATCH_TRAIN_ACC)
        print(f"avg train loss: {avg_train_loss:>7f},  avg train acc: {avg_train_acc:>7f}")
        DA_EPOCH_TRAIN_LOSS.append(avg_train_loss)
        DA_EPOCH_TRAIN_ACC.append(avg_train_acc)
            
        """ test """ 
        MODEL.eval()
        MODEL.to(DEVICE)
        size = len(sketch_test_loader.dataset)
        with torch.no_grad():
            for batch, (X, y) in enumerate(sketch_test_loader):
                # X shape: (B:64, H:224, W:224, C:3) -> (B:64, C:3, H:224, W:224)
                # y: an int from [0, ..., 9]
                X, y = torch.from_numpy(np.transpose(X.numpy().astype('float32'), (0, 3, 1, 2))).to(DEVICE), y.to(DEVICE)
                pred = MODEL(X)
                # cross-entropy loss from this batch
                loss = CLS_LOSS_FN(pred, y)
                loss = loss.item()
                DA_BATCH_TEST_LOSS.append(loss)
                # accuracy from this batch
                acc =  (pred.argmax(1) == y).type(torch.float).sum().item() / len(X)
                DA_BATCH_TEST_ACC.append(acc)
                # print results
                if batch % 50 == 0:
                    current = batch * sketch_test_loader.batch_size + len(X)
                    print(f"test loss: {loss:>7f}, acc: {acc:>7f}    [{current:>5d} / {size:>5d}]")
            avg_test_loss = np.mean(DA_BATCH_TEST_LOSS)
            avg_test_acc = np.mean(DA_BATCH_TEST_ACC)
            print(f"avg test loss: {avg_test_loss:>7f},  avg test acc: {avg_test_acc:>7f}")
            DA_EPOCH_TEST_LOSS.append(avg_test_loss)
            DA_EPOCH_TEST_ACC.append(avg_test_acc)
            

    plot_metric(DA_EPOCH_TRAIN_LOSS, DA_EPOCH_TRAIN_ACC,
                DA_EPOCH_TEST_LOSS, DA_EPOCH_TEST_ACC,
                title="Domain Adaption")