# config.py
# describe all the parameters

class Config:
    # datas
    data_root = "./data"
    num_classes = 10
    image_size = 32

    # train
    batch_size = 128
    epochs = 10
    lr = 1e-3
    weight_decay = 1e-4
    num_workers = 2
    seed = 42

    # save
    ckpt_dir = "./checkpoints"
    log_dir = "./runs"
    model_name = "simple_cnn"

    # device
    device = "cuda"

cfg = Config()