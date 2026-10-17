# Hyperparameters


class Config:
    # data
    data_root = "../../data"
    num_train_pairs = 60000
    num_val_pairs = 10000
    positive_ratio = 0.5

    # model
    embedding_dim = 128

    # loss
    margin = 1.0

    # train
    batch_size = 128
    lr = 1e-3
    weight_decay = 1e-5
    epochs = 30
    num_workers = 2
    seed = 42
    log_interval = 100

    # paths
    ckpt_dir = "../checkpoints"
    