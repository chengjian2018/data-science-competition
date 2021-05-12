# import os
# import json
# import argparse
# import torch
# import dataloaders
# import models
# import inspect
# import math
# from utils import losses
# from utils import Logger
# from utils.torchsummary import summary
# from trainer import Trainer

# def get_instance(module, name, config, *args):
#     # GET THE CORRESPONDING CLASS / FCT
#     return getattr(module, config[name]['type'])(*args, **config[name]['args'])
#     # DATA LOADERS
# if __name__=='__main__':
#     parser = argparse.ArgumentParser(description='PyTorch Training')
#     parser.add_argument('-c', '--config', default='config.json',type=str,
#                         help='Path to the config file (default: config.json)')
#     parser.add_argument('-r', '--resume', default=None, type=str,
#                         help='Path to the .pth model checkpoint to resume training')
#     parser.add_argument('-d', '--device', default=None, type=str,
#                            help='indices of GPUs to enable (default: all)')
#     args = parser.parse_args()
#
#     config = json.load(open(args.config))
#     if args.resume:
#         config = torch.load(args.resume)['config']
#     if args.device:
#         os.environ["CUDA_VISIBLE_DEVICES"] = args.device
#     train_loader = get_instance(dataloaders, 'train_loader', config)
#     val_loader = get_instance(dataloaders, 'val_loader', config)
#     model = get_instance(models, 'arch', config, train_loader.dataset.num_classes)
#     print(f'\n{model}\n')
#     print(train_loader.dataset.num_classes)

cfg ={
    'filename' :'1.tif',
    'provider':'DUT',
    'author':'CJ',
    'pluginname':'地物标注',
    'pluginclass':'标注',
    'resultfile':'1_gt.png'
}

# from xml.dom.minidom import Document
#
# def create_xml(cfg,path):
#     f = open(path,'w',encoding='utf-8')
#     f.write(Document().toprettyxml(indent=" "))
#     f.writelines('<annotation>\n')
#     f.writelines('              <source>\n')
#     f.writelines('                       <filename>%s</filename>\n'%str(cfg['filename']))
#     f.writelines('                       <origin>GF2/GF3</ororigin>\n')
#     f.writelines('              </source>\n')
#     f.writelines('              <research>\n')
#     f.writelines('                        <version>4.0</version>\n')
#     f.writelines('                        <provider>%s</provider>\n'%str(cfg['provider']))
#     f.writelines('                        <author>%s</author>\n'%str(cfg['author']))
#     f.writelines('                        <pluginname>%s</pluginname>\n'%str(cfg['pluginname']))
#     f.writelines('                        <pluginclass>%s<\pluinclass>\n'%str(cfg['pluginclass']))
#     f.writelines('                        <time>2020-07-2020-11</time>\n')
#     f.writelines('               </research>\n')
#     f.writelines('               <segmentation>\n')
#     f.writelines('                        <resultfile>%s</resultfile>\n'%str(cfg['resultfile']))
#     f.writelines('               </segmentation>\n')
#     f.writelines('</annotation>')
#     f.close()

import xml.etree.ElementTree as ET
import sys
def change_xml(inp,out):
    doc = ET.parse(inp)
    root = doc.getroot()
    # import pdb
    # pdb.set_trace()
    sub1 = root.find('source').find('filename')
    sub2 = root.find('research').find('provider')
    sub3 = root.find('research').find('author')
    sub4 = root.find('research').find('pluginname')
    sub5 = root.find('research').find('pluginclass')
    sub6 = root.find('segmentation').find('resultfile')
    sub4.text = str(cfg['pluginname'])
    sub3.text = str(cfg['author'])
    sub1.text = str(cfg['filename'])
    sub2.text = str(cfg['provider'])
    sub5.text = str(cfg['pluginclass'])
    sub6.text = str(cfg['resultfile'])
    doc.write(out)

if __name__ == '__main__':
    change_xml(sys.argv[1],sys.argv[2])