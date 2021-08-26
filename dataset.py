import numbers
import os
import queue as Queue
import threading

import mxnet as mx
import numpy as np
import torch, cv2
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class BackgroundGenerator(threading.Thread):
    def __init__(self, generator, local_rank, max_prefetch=6):
        super(BackgroundGenerator, self).__init__()
        self.queue = Queue.Queue(max_prefetch)
        self.generator = generator
        self.local_rank = local_rank
        self.daemon = True
        self.start()

    def run(self):
        torch.cuda.set_device(self.local_rank)
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def next(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

    def __next__(self):
        return self.next()

    def __iter__(self):
        return self


class DataLoaderX(DataLoader):

    def __init__(self, local_rank, debug=False, **kwargs):
        super(DataLoaderX, self).__init__(**kwargs)
        self.stream = torch.cuda.Stream(local_rank)
        self.local_rank = local_rank
        self.debug = debug
        if self.debug:
            self.fix_batch = next(super(DataLoaderX, self).__iter__(), None)
            for k in range(len(self.fix_batch)):
                self.fix_batch[k] = self.fix_batch[k].to(device=self.local_rank, non_blocking=True)

    def __iter__(self):
        self.iter = super(DataLoaderX, self).__iter__()
        self.iter = BackgroundGenerator(self.iter, self.local_rank)
        self.preload()
        return self

    def preload(self):
        self.batch = next(self.iter, None)
        if self.batch is None:
            return None
        with torch.cuda.stream(self.stream):
            for k in range(len(self.batch)):
                self.batch[k] = self.batch[k].to(device=self.local_rank, non_blocking=True)

    def __next__(self):
        if self.debug:
            return self.fix_batch
        else:
            torch.cuda.current_stream().wait_stream(self.stream)
            batch = self.batch
            if batch is None:
                raise StopIteration
            self.preload()
            return batch


class MXFaceDataset(Dataset):
    def __init__(self, root_dir, local_rank):
        super(MXFaceDataset, self).__init__()
        self.transform = transforms.Compose(
            [transforms.ToPILImage(),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
             ])
        self.root_dir = root_dir
        self.local_rank = local_rank
        path_imgrec = os.path.join(root_dir, 'train.rec')
        path_imgidx = os.path.join(root_dir, 'train.idx')
        self.imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')
        s = self.imgrec.read_idx(0)
        header, _ = mx.recordio.unpack(s)
        if header.flag > 0:
            self.header0 = (int(header.label[0]), int(header.label[1]))
            self.imgidx = np.array(range(1, int(header.label[0])))
        else:
            self.imgidx = np.array(list(self.imgrec.keys))

    def __getitem__(self, index):
        idx = self.imgidx[index]
        s = self.imgrec.read_idx(idx)
        header, img = mx.recordio.unpack(s)
        label = header.label
        if not isinstance(label, numbers.Number):
            label = label[0]
        label = torch.tensor(label, dtype=torch.long)
        sample = mx.image.imdecode(img).asnumpy()
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, label

    def __len__(self):
        return len(self.imgidx)


class ECFaceDataset(Dataset):
  def __init__(self, root_dir):
    self.transform = transforms.Compose(
        [transforms.ToPILImage(),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),                                 
    ])

    self.idx_imgs = []
    self.idx_labels = []

    labels = os.listdir(root_dir)

    for label in labels:
      path_label_imgs = os.path.join(root_dir, label)
      for img_name in os.listdir(path_label_imgs):
        path_img = os.path.join(path_label_imgs, img_name)
        img = cv2.imread(path_img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.idx_imgs.append(img)
        self.idx_labels.append(label)

  def __len__(self):
    return len(self.idx_labels)

  def __getitem__(self, index):
    idx_label = self.idx_labels[index]
    idx_img = self.idx_imgs[index]

    if self.transform is not None:
      idx_img = self.transform(idx_img)

    return idx_img, idx_label
