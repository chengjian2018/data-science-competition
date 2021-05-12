import os
from PIL import Image
import numpy as np
import skimage.io as io
from tqdm import tqdm
import PIL
import cv2
from segnet.utils import palette as pl
import matplotlib.pyplot as plt
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
def bmptopng(file_path,suffix):
    files = recursive_glob(rootdir=file_path,suffix=suffix)
    for filename in files:
        newfilename = filename[0:filename.find(".")]+".png"
        print(newfilename)
        img = Image.open(filename)
        img.save(newfilename)
        os.remove(filename)

def img_reshape(root):
    image_dir = os.path.join(root, 'images')
    label_dir = os.path.join(root, 'masks')
    files = recursive_glob(rootdir=image_dir, suffix='.png')
    if not os.path.exists(os.path.join(image_dir,'reshape')):
        os.makedirs(os.path.join(image_dir,'reshape'))
    if not os.path.exists(os.path.join(label_dir,'reshape')):
        os.makedirs(os.path.join(label_dir,'reshape'))
    if not files:
        raise Exception("No files found in %s"%(image_dir))
    print("Found %d images in %s"%(len(files),image_dir))
    for index in tqdm(range(len(files))):
        image_path = files[index].rstrip()
        label_path = os.path.join(
            label_dir,
            image_path.split(os.sep)[-1],
        )
        img = io.imread(image_path)
        lbl = io.imread(label_path)
        img_out,lbl_out = np.zeros((380,384,3)), np.zeros((380,384))
        for i in range(288):
            # import pdb
            # pdb.set_trace()
            img_out[i,:,:] = img[i,:,:]
            lbl_out[i,:] = lbl[i,:]
        for j in range(16):
            img_out[288+int(j*5.4118):288+int((j+1)*5.4118),:,:] = (j+1)*15
            lbl_out[288+int(j*5.4118):288+int((j+1)*5.4118),:] = j+5
        img_out[375:,:,:] = 255
        lbl_out[375:,:] = 21
        img_np = Image.fromarray(img_out.astype(np.uint8))
        lbl_np = Image.fromarray(lbl_out.astype(np.uint8))
        img_np.save(os.path.join(image_dir,'reshape',image_path.split(os.sep)[-1]))
        lbl_np.save(os.path.join(label_dir, 'reshape', image_path.split(os.sep)[-1]))

def lbl_load(root):
    label_dir = os.path.join(root, 'masks')
    images_dir = os.path.join(root, 'images')
    files = recursive_glob(rootdir=images_dir, suffix='.png')
    if not os.path.exists(os.path.join(label_dir,'colored')):
        os.makedirs(os.path.join(label_dir,'colored'))
    if not files:
        raise Exception("No files found in %s"%(label_dir))
    print("Found %d images in %s"%(len(files),label_dir))
    for index in tqdm(range(len(files))):
        image_path = files[index].rstrip()
        label_path = os.path.join(
            label_dir,
            image_path.split(os.sep)[-1],
        )
        img = io.imread(label_path)
        mask = colorize_mask(img)
        mask.save(os.path.join(label_dir,'colored',label_path.split(os.sep)[-1]))

# def pic_display():
#     v4 = ['1','4','6','16']
#     v3 = ['11','12','14','15']
#     f, axarr = plt.subplots(4, 4,sharex=True,sharey=True)
#     f.subplots_adjust(wspace=0,hspace=0)
#     axarr[0][0].set_title('v3')
#     axarr[0][1].set_title('v3_box')
#     axarr[0][2].set_title('v4')
#     axarr[0][3].set_title('v4_box')
#     for j in range(4):
#         axarr[j][0].imshow(io.imread(os.path.join('h:/dataset/results',v3[j]+'_bbox.png')))
#         axarr[j][1].imshow(io.imread(os.path.join('h:/dataset/results',v3[j]+'_rebbox.png')))
#         axarr[j][2].imshow(io.imread(os.path.join('h:/dataset/results',v4[j]+'_bbox.png')))
#         axarr[j][3].imshow(io.imread(os.path.join('h:/dataset/results',v4[j]+'_rebbox.png')))
#     plt.show()

# def pic_display():
#     v4 = ['1','4','6','16']
#     v3 = ['11','12','14','15']
#     f, axarr = plt.subplots(4, 4,sharex=True,sharey=True)
#     f.subplots_adjust(wspace=0,hspace=0)
#     axarr[0][0].set_title('origin')
#     axarr[0][1].set_title('fcn8')
#     axarr[0][2].set_title('pspnet')
#     axarr[0][3].set_title('unet')
#     for j in range(4):
#         axarr[j][0].imshow(io.imread(os.path.join('h:/dataset/results',str(j+1)+'.png')))
#         axarr[j][1].imshow(io.imread(os.path.join('h:/dataset/results',str(j+1)+'-fcn8.png')))
#         axarr[j][2].imshow(io.imread(os.path.join('h:/dataset/results',str(j+1)+'-pspnet.png')))
#         axarr[j][3].imshow(io.imread(os.path.join('h:/dataset/results',str(j+1)+'-unet.png')))
#     plt.show()
# def pic_display():
#     CVC1 = ['CVC_1.png','CVC_2.png','CVC_3.png','CVC_4.png']
#     CVC2 = ['CVC_5.png','CVC_6.png','CVC_7.png','CVC_8.png']
#     LBL1 = ['LBL_1.png','LBL_2.png','LBL_3.png','LBL_4.png']
#     LBL2 = ['LBL_5.png','LBL_6.png','LBL_7.png','LBL_8.png']
#     ETIS =['ETIS_1.tif','ETIS_2.tif','ETIS_3.tif','ETIS_4.tif']
#     f, axarr = plt.subplots(4, 5,sharex=True,sharey=True)
#     f.subplots_adjust(wspace=0,hspace=0)
#     axarr[0][0].set_title('CLINIC')
#     axarr[0][1].set_title('LABEL')
#     axarr[0][2].set_title('COLON')
#     axarr[0][3].set_title('LABEL')
#     axarr[0][4].set_title('ETIS')
#     for j in range(4):
#         colon = io.imread(os.path.join('h:/dataset/results',CVC2[j]))
#         colon = cv2.resize(colon,(1000,1000))
#         colon_l = io.imread(os.path.join('h:/dataset/results',LBL2[j]))
#         colon_l = cv2.resize(colon_l,(1000,1000))
#         clinic = io.imread(os.path.join('h:/dataset/results',CVC1[j]))
#         clinic = cv2.resize(clinic,(1000,1000))
#         clinic_l = io.imread(os.path.join('h:/dataset/results',LBL1[j]))
#         clinic_l = cv2.resize(clinic_l,(1000,1000))
#         etis = io.imread(os.path.join('h:/dataset/results',ETIS[j]))
#         etis = cv2.resize(etis,(1000,1000))
#         axarr[j][0].imshow(clinic)
#         axarr[j][1].imshow(clinic_l)
#         axarr[j][2].imshow(colon)
#         axarr[j][3].imshow(colon_l)
#         axarr[j][4].imshow(etis)
#     plt.show()
def pic_display():
    v4 = ['s1.jpg','s2.jpg','s3.jpg','s4.jpg','s5.jpg']
    v3 = ['s1.png','s2.png','s3.png','s4.png','s5.png']
    f, axarr = plt.subplots(2, 5,sharex=True,sharey=True)
    f.subplots_adjust(wspace=0,hspace=0)
    for j in range(5):
        a = io.imread(os.path.join('h:/dataset/results',v4[j]))
        a = cv2.resize(a,(1000,1500))
        b = io.imread(os.path.join('h:/dataset/results',v3[j]))
        b = cv2.resize(b,(1000,1500))
        axarr[0][j].imshow(a)
        axarr[1][j].imshow(b)
    plt.show()
# def pic_display():
#     v3_ = ['v3_80.tif','v3_90.tif','v3_155.bmp','v3_162.bmp']
#     v4_ = ['v4_80.png', 'v4_90.png', 'v4_155.png', 'v4_162.png']
#     path = 'h:/dataset/results'
#     f, axarr = plt.subplots(2, 4,sharex=True,sharey=True)
#     f.subplots_adjust(wspace=0,hspace=0)
#     for j in range(4):
#         v3 = cv2.imread(os.path.join(path,v3_[j]))
#         v4 = cv2.imread(os.path.join(path,v4_[j]))
#         v3 = cv2.cvtColor(v3,cv2.COLOR_BGR2RGB)
#         v4 = cv2.cvtColor(v4, cv2.COLOR_BGR2RGB)
#         v3 = cv2.resize(v3,(1000,1000))
#         v4 = cv2.resize(v4, (1000, 1000))
#         axarr[0][j].imshow(v3)
#         axarr[1][j].imshow(v4)
#     plt.show()
def CVC(path,split):
    if split == 'CVC-612':
        w,h = 384,288
        file_path = os.path.join(path,split)
        out_path = os.path.join(file_path,'label')
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        for i in tqdm(range(1,613)):
            lumen = io.imread(os.path.join(file_path,'gtlumen',str(i)+'.bmp'))
            border = io.imread(os.path.join(file_path,'border',str(i)+'.bmp'))
            polyp = io.imread(os.path.join(file_path,'gtpolyp',str(i)+'.tif'))
            lbl = np.zeros((h,w))
            lbl[border > 0] = 1
            lbl[lumen>0] = 3
            lbl[polyp>0] = 2
            mask = colorize_mask(lbl)
            mask.save(os.path.join(out_path,str(i)+'.png'))
    else:
        w, h = 574, 500
        file_path = os.path.join(path, split)
        out_path = os.path.join(file_path, 'label')
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        for i in tqdm(range(1, 301)):
            lumen = io.imread(os.path.join(file_path, 'gtlumen', str(i) + '.bmp'))
            border = io.imread(os.path.join(file_path, 'border', str(i) + '.bmp'))
            polyp = io.imread(os.path.join(file_path, 'gtpolyp', str(i) + '.bmp'))
            lbl = np.zeros((h, w))
            lbl[border >0] = 1
            lbl[lumen >0] = 3
            lbl[polyp >0] = 2
            mask = colorize_mask(lbl)
            mask.save(os.path.join(out_path,str(i+612)+'.png'))

def chagecolor(path):
    if not os.path.exists(os.path.join(path,'color')):
        os.makedirs(os.path.join(path,'color'))
    for i in range(613,913):
        img = cv2.imread(os.path.join(path,str(i)+'.png'))
        # img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img = PIL.Image.fromarray(img.astype(np.uint8))
        img.save(os.path.join(path,'color',str(i)+'.png'))

def reshape(path):
    for i in tqdm(range(1,913)):
        img = cv2.imread(os.path.join(path,'images',str(i)+'.png'))
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img = cv2.resize(img,(400,400))
        img = PIL.Image.fromarray(img.astype(np.uint8))
        img.save(os.path.join(path,'images','reshape',str(i)+'.png'))
        lbl =  cv2.imread(os.path.join(path,'masks',str(i)+'.png'))
        lbl = cv2.cvtColor(lbl, cv2.COLOR_BGR2RGB)
        lbl = cv2.resize(lbl,(400,400))
        lbl = PIL.Image.fromarray(lbl.astype(np.uint8)).convert('P')
        lbl.save(os.path.join(path,'masks','reshape',str(i)+'.png'))
import os.path as osp

# def colorize_mask(temp):
#     label_colours = pl.sat_palette
#     r = temp.copy()
#     g = temp.copy()
#     b = temp.copy()
#     for l in range(9):
#         r[temp == l] = label_colours[l][0]
#         g[temp == l] = label_colours[l][1]
#         b[temp == l] = label_colours[l][2]
#
#     rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
#     rgb[:, :, 0] = r  # / 255.0
#     rgb[:, :, 1] = g  # / 255.0
#     rgb[:, :, 2] = b  # / 255.0
#     rgb = Image.fromarray(rgb.astype(np.uint8))
#     return rgb
def colorize_mask(mask, palette):
    zero_pad = 256 * 3 - len(palette)
    for i in range(zero_pad):
                    palette.append(0)
    new_mask = PIL.Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask

def color2lbl():
    path = 'j:/data'
    color2label = {
        '[0 0 0]': 0,
        '[  0   0 255]': 1,
        '[  0 255   0]': 2,
        '[  0 128   0]': 3,
        '[  0 255 255]': 4,
        '[255 255   0]': 5,
        '[255 255 255]': 6,
        '[255   0 255]': 7,
        '[128 128 128]': 8
    }
    for k in tqdm(range(174, 501)):
        label = io.imread(osp.join(path, str(k) + '_gt.png'))
        rgb_label = label[:, :, :-1]
        lbl = np.zeros((512, 512))
        for i in range(512):
            for j in range(512):
                lbl[i, j] = color2label[str(rgb_label[i, j, :])]
        lbl = colorize_mask(lbl)
        lbl.save(osp.join(path, 'lbl', str(k) + '_gt.png'))
def lbl_colored():
    path = 'j:/lbl'
    out = 'j:/colored'
    pa = pl.sat_palette
    for k in tqdm(range(1,501)):
        label = io.imread(osp.join(path,str(k) + '_gt.png'))
        lbl = colorize_mask(label,pa)
        lbl.save(osp.join(out,str(k) + '_gt.png'))

if __name__ == '__main__':
    lbl_colored()
    # color2lbl()
    # file_path = 'h:/dataset/cvc-912'
    # reshape(file_path)
    # pic_display()
    # lbl_load(file_path)
    # bmptopng(file_path,'.bmp')
    # splits = 'train'
    # lbl_load(os.path.join('h:/dataset/cvc',splits))
    # pic_display()
    # path = 'J:/CVC-EndoSceneStill'
    # splits = ['CVC-612','CVC-300']
    # for split in splits:
    #     CVC(path,split)
    # for i in range(4,300):
    #     img = io.imread(os.path.join(path,'CVC-300','gtspecular',str(i)+'.bmp'))
    #     print(img.shape)
    # in_path = os.path.join(path,'CVC-300','bbdd')
    # out_path = os.path.join(path, 'CVC-612', 'bbdd')
    # for i in tqdm(range(1,301)):
    #     img = cv2.imread(os.path.join(in_path,str(i)+'.bmp'))
    #     img = PIL.Image.fromarray(img.astype(np.uint8)).convert('P')
    #     img.save(os.path.join(out_path,str(i+612)+'.bmp'))

