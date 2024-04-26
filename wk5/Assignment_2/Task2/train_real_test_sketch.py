from utils import *
import numpy as np
from torch.optim import Adam, SGD
from torch.nn import CrossEntropyLoss


if __name__ == '__main__':
    EPOCHS = 10
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-5
    MODEL = ResNet34DomainNet()
    # OPTIMISER = SGD(MODEL.parameters(), LEARNING_RATE)
    OPTIMISER = Adam(MODEL.parameters(), LEARNING_RATE)
    LOSS_FN = CrossEntropyLoss()
    
    real_train_loader = get_dataloader(REAL_TRAIN_PATH, BATCH_SIZE)
    sketch_test_loader = get_dataloader(SKETCH_TEST_PATH, BATCH_SIZE)
        
    for epoch in range(EPOCHS):
        print(f"--------------- epoch {epoch+1} ---------------")
        train_loop(real_train_loader, MODEL, LOSS_FN, OPTIMISER)
        test_loop(sketch_test_loader, MODEL, LOSS_FN)
        
    plot_metric(EPOCH_TRAIN_LOSS, EPOCH_TRAIN_ACC, 
                EPOCH_TEST_LOSS, EPOCH_TEST_ACC,
                title="real_train & sketch_test")