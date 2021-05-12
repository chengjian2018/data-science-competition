from base import BaseDataSet, BaseDataLoader
from utils import palette
# from segnet.base import BaseDataSet, BaseDataLoader
# from segnet.utils import palette
import numpy as np
import os
import scipy
import torch
from PIL import Image
import cv2
from torch.utils.data import Dataset
from torchvision import transforms

def recursive_glob(rootdir=".", suffix=""):
    """Performs recursive glob with given suffix and rootdir
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    return [
        os.path.join(looproot, filename)
        for looproot, _, filenames in os.walk(rootdir)
        for filename in filenames
        if filename.endswith(suffix)
    ]

class VOCDataset(BaseDataSet):
    """
    Pascal Voc dataset
    http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
    """
    def __init__(self, **kwargs):
        self.num_classes = 9
        self.palette = palette.sat_palette
        super(VOCDataset, self).__init__(**kwargs)

    def _set_files(self):
        self.image_dir = os.path.join(self.root, 'img')
        self.label_dir = os.path.join(self.root, 'lbl')

        self.files = recursive_glob(rootdir=self.image_dir,suffix='.tif')

    def _load_data(self, index):
        image_path = self.files[index].rstrip()
        image_id = image_path.split("\\")[-1].split(".")[0]
        label_path = os.path.join(
            self.label_dir,
            str(image_id)+'_gt.png',
        )
        image = np.asarray(Image.open(image_path), dtype=np.float32)
        label = np.asarray(Image.open(label_path), dtype=np.int32)
        return image, label ,image_id

class VOCAugDataset(BaseDataSet):
    """
    Contrains both SBD and VOC 2012 dataset
    Annotations : https://github.com/DrSleep/tensorflow-deeplab-resnet#evaluation
    Image Sets: https://ucla.app.box.com/s/rd9z2xvwsfpksi7mi08i2xqrj7ab4keb/file/55053033642
    """
    def __init__(self, **kwargs):
        self.num_classes = 9
        self.palette = palette.get_voc_palette(self.num_classes)
        super(VOCAugDataset, self).__init__(**kwargs)

    def _set_files(self):
        self.root = os.path.join(self.root, 'VOCdevkit/VOC2012')

        file_list = os.path.join(self.root, "ImageSets/Segmentation", self.split + ".txt")
        file_list = [line.rstrip().split(' ') for line in tuple(open(file_list, "r"))]
        self.files, self.labels = list(zip(*file_list))

    def _load_data(self, index):
        image_path = os.path.join(self.root, self.files[index][1:])
        label_path = os.path.join(self.root, self.labels[index][1:])
        image = np.asarray(Image.open(image_path), dtype=np.float32)
        label = np.asarray(Image.open(label_path), dtype=np.int32)
        image_id = self.files[index].split("/")[-1].split(".")[0]
        return image, label, image_id


class  SAT(BaseDataLoader):
    def __init__(self, data_dir, batch_size, split, crop_size=None, base_size=None, scale=True, num_workers=1, val=False,
                    shuffle=False, flip=False, rotate=False, blur= False, augment=False, val_split= None, return_id=False):

        self.MEAN = [0.45734706, 0.43338275, 0.40058118]
        self.STD = [0.23965294, 0.23532275, 0.2398498]

        kwargs = {
            'root': data_dir,
            'split': split,
            'mean': self.MEAN,
            'std': self.STD,
            'augment': augment,
            'crop_size': crop_size,
            'base_size': base_size,
            'scale': scale,
            'flip': flip,
            'blur': blur,
            'rotate': rotate,
            'return_id': return_id,
            'val': val
        }

        if split in ["train_aug", "trainval_aug", "val_aug", "test_aug"]:
            self.dataset = VOCAugDataset(**kwargs)
        elif split in ["train", "trainval", "val", "test"]:
            self.dataset = VOCDataset(**kwargs)
        else: raise ValueError(f"Invalid split name {split}")
        super(SAT, self).__init__(self.dataset, batch_size, shuffle, num_workers, val_split)

def get_instance( name, config, *args):
    # GET THE CORRESPONDING CLASS / FCT
    return SAT(*args, **config[name]['args'])

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import json
    config = json.load(open('h:/cj/segnet/config_pspnet.json'))
    train_loader = get_instance('train_loader', config)
    for i, data_samples in enumerate(train_loader):
        imgs, labels = data_samples
        # import pdb
        #         #
        #         # pdb.set_trace()
        imgs = imgs.numpy()[:, ::-1, :, :]#
        imgs = np.transpose(imgs, [0,2,3,1])
        f, axarr = plt.subplots(4, 2)
        for j in range(4):
            axarr[j][0].imshow(imgs[j])
            axarr[j][1].imshow(labels.numpy()[j])
        plt.show()
        a = input()
        if a == "ex":
            break
        else:
            plt.close()