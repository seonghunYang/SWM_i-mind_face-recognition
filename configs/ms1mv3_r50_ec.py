from easydict import EasyDict as edict

# make training faster
# our RAM is 256G
# mount -t tmpfs -o size=140G  tmpfs /train_tmp

config = edict()
config.loss = "arcface"
config.network = "r50"
config.resume = True
config.savefolder = "./model/checkpoint"
config.output = "./model/ms1m_arcface/"
config.embedding_size = 512
config.sample_rate = 1.0
config.fp16 = True
config.momentum = 0.9
config.weight_decay = 5e-4
config.batch_size = 3
config.lr = 0.1  # batch size is 512

config.rec = "../our_children_changed/dataset"
config.verpath = "./model/check/"
config.num_classes = 10
config.num_image = 392
config.num_epoch = 25
config.warmup_epoch = -1
config.decay_epoch = [10, 16, 22]
config.val_targets = ["lfw", "cfp_fp", "agedb_30"]