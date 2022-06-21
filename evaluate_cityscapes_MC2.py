import argparse
import scipy
from scipy import ndimage
import numpy as np
import sys
from packaging import version
from multiprocessing import Pool
import torch
from torch.autograd import Variable
import torchvision.models as models
import torch.nn.functional as F
from torch.utils import data, model_zoo
from model.deeplab import Res_Deeplab
from model.deeplab_multi import DeeplabMulti
from model.deeplab_vgg import DeeplabVGG
from dataset.cityscapes_dataset import cityscapesDataSet
from collections import OrderedDict
import os
from PIL import Image
from utils.tool import fliplr
import matplotlib.pyplot as plt
import torch.nn as nn
import yaml
import time


torch.backends.cudnn.benchmark=True

IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

DATA_DIRECTORY = './data/Cityscapes/data'
DATA_LIST_PATH = './dataset/cityscapes_list/val.txt'
SAVE_PATH = './result/cityscapes_MC'


IGNORE_LABEL = 255
NUM_CLASSES = 19
max_entropy = np.log(NUM_CLASSES)
NUM_STEPS = 500 # Number of images in the validation set.
RESTORE_FROM = 'http://vllab.ucmerced.edu/ytsai/CVPR18/GTA2Cityscapes_multi-ed35151c.pth'
RESTORE_FROM_VGG = 'http://vllab.ucmerced.edu/ytsai/CVPR18/GTA2Cityscapes_vgg-ac4ac9f6.pth'
RESTORE_FROM_ORC = 'http://vllab1.ucmerced.edu/~whung/adaptSeg/cityscapes_oracle-b7b9934.pth'
SET = 'val'

MODEL = 'DeeplabMulti'

palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)



def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)

    return new_mask

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--model", type=str, default=MODEL,
                        help="Model Choice (DeeplabMulti/DeeplabVGG/Oracle).")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the Cityscapes dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--gpu", type=int, default=0,
                        help="choose gpu device.")
    parser.add_argument("--batchsize", type=int, default=16,
                        help="choose gpu device.")
    parser.add_argument("--set", type=str, default=SET,
                        help="choose evaluation set.")
    parser.add_argument("--save", type=str, default=SAVE_PATH,
                        help="Path to save result.")
    return parser.parse_args()

def save(output_name):
    output, name = output_name
    output_col = colorize_mask(output)
    output = Image.fromarray(output)

    output.save('%s' % (name))
    output_col.save('%s_color.png' % (name.split('.jpg')[0]))
    return

def save_heatmap(output_name):
    output, name = output_name
    fig = plt.figure()
    plt.axis('off')
    heatmap = plt.imshow(output, cmap='viridis')
    #fig.colorbar(heatmap)
    fig.savefig('%s_heatmap.png' % (name.split('.jpg')[0]))
    return

def save_scoremap(output_name):
    output, name = output_name
    fig = plt.figure()
    plt.axis('off')
    heatmap = plt.imshow(output, cmap='viridis')
    #fig.colorbar(heatmap)
    fig.savefig('%s_scoremap.png' % (name.split('.jpg')[0]))
    return

def main():
    """Create the model and start the evaluation process."""
    args = get_arguments()

    config_path = os.path.join(os.path.dirname(args.restore_from),'opts.yaml')
    with open(config_path, 'r') as stream:
        config = yaml.load(stream)

    args.model = config['model']
    print('ModelType:%s'%args.model)
    print('NormType:%s'%config['norm_style'])
    gpu0 = args.gpu
    batchsize = args.batchsize

    model_name = os.path.basename( os.path.dirname(args.restore_from) )
    args.save += model_name

    if not os.path.exists(args.save):
        os.makedirs(args.save)

    if args.model == 'DeepLab':
        model = DeeplabMulti(num_classes=args.num_classes, use_se = config['use_se'], train_bn = False, norm_style = config['norm_style'])
    elif args.model == 'Oracle':
        model = Res_Deeplab(num_classes=args.num_classes)
        if args.restore_from == RESTORE_FROM:
            args.restore_from = RESTORE_FROM_ORC
    elif args.model == 'DeeplabVGG':
        model = DeeplabVGG(num_classes=args.num_classes)
        if args.restore_from == RESTORE_FROM:
            args.restore_from = RESTORE_FROM_VGG

    if args.restore_from[:4] == 'http' :
        saved_state_dict = model_zoo.load_url(args.restore_from)
    else:
        saved_state_dict = torch.load(args.restore_from)

    try:
        model.load_state_dict(saved_state_dict)
    except:
        model = torch.nn.DataParallel(model)
        model.load_state_dict(saved_state_dict)
    #model = torch.nn.DataParallel(model)
    model.eval()
    model.cuda(gpu0)

    testloader = data.DataLoader(cityscapesDataSet(args.data_dir, args.data_list, crop_size=(512, 1024), resize_size=(1024, 512), mean=IMG_MEAN, scale=False, mirror=False, set=args.set),
                                    batch_size=batchsize, shuffle=False, pin_memory=True, num_workers=4)

    scale = 1.25
    testloader2 = data.DataLoader(cityscapesDataSet(args.data_dir, args.data_list, crop_size=(round(512*scale), round(1024*scale) ), resize_size=( round(1024*scale), round(512*scale)), mean=IMG_MEAN, scale=False, mirror=False, set=args.set),
                                    batch_size=batchsize, shuffle=False, pin_memory=True, num_workers=2)


    if version.parse(torch.__version__) >= version.parse('0.4.0'):
        interp = nn.Upsample(size=(1024, 2048), mode='bilinear', align_corners=True)
    else:
        interp = nn.Upsample(size=(1024, 2048), mode='bilinear')

    sm = torch.nn.Softmax(dim = 1)
    log_sm = torch.nn.LogSoftmax(dim = 1)
    kl_distance = nn.KLDivLoss( reduction = 'none')

    for index, img_data in enumerate(zip(testloader, testloader2)):
        batch, batch2 = img_data
        image, _, _, name = batch
        batch_size = image.size(0)
        # h = image.size(2)
        # w = image.size(3)
        inputs = image.cuda()
        
        image2, _, _, name2 = batch2
        inputs2 = image2.cuda()
        print('\r>>>>Extracting feature...%03d/%03d'%(index*batchsize, NUM_STEPS), end='')
        #image3, _, _, name3 = batch3
        M = 8
        M_float = float(M)
        p = torch.zeros(batch_size,19, 1024, 2048).cuda() # (shape: (batch_size, num_classes, h, w))
        for i in range(M):
            with torch.no_grad():
                output1, output2 = model(inputs)
                output_batch = interp(sm(0.5* output1 + output2))
                if (i == 1) :
                 heatmap_output1, heatmap_output2 = output1, output2
                #output_batch = interp(sm(output1))
                #output_batch = interp(sm(output2))
                output1, output2 = model(fliplr(inputs))
                output1, output2 = fliplr(output1), fliplr(output2)
                output_batch += interp(sm(0.5 * output1 + output2))
                if (i == 1) :
                    heatmap_output1, heatmap_output2 = heatmap_output1+output1, heatmap_output2+output2
                #output_batch += interp(sm(output1))
                #output_batch += interp(sm(output2))
                del output1, output2, inputs

                output1, output2 = model(inputs2)
                output_batch += interp(sm(0.5* output1 + output2))
                #output_batch += interp(sm(output1))
                #output_batch += interp(sm(output2))
                output1, output2 = model(fliplr(inputs2))
                output1, output2 = fliplr(output1), fliplr(output2)
                output_batch += interp(sm(0.5 * output1 + output2))
                #output_batch += interp(sm(output1))
                #output_batch += interp(sm(output2))
                del output1, output2, inputs2
                #output_batch = output_batch.cpu().data.numpy()
                if(i == 1) :
                    heatmap_batch = torch.sum(kl_distance(log_sm(heatmap_output1), sm(heatmap_output2)), dim=1) 
                    heatmap_batch = torch.log(1 + 10*heatmap_batch) # for visualization
                    heatmap_batch = heatmap_batch.cpu().data.numpy()
            # _, logits_downsampled = model(Variable(inputs).cuda()) # (shape: (batch_size, num_classes, h/8, w/8))
            # logits = F.upsample(input=logits_downsampled , size=(h, w), mode='bilinear', align_corners=True) # (shape: (batch_size, num_classes, h, w))
            # p_value = F.softmax(logits, dim=1) # (shape: (batch_size, num_classes, h, w))
                p = p + output_batch/M_float

        output_batch = p.transpose(0, 2, 3, 1) # (array of shape: (batch_size, h, w, num_classes))
        #output_batch = output_batch.transpose(0,2,3,1)
        #output_batch = output_batch.transpose(0,2,3,1)
        scoremap_batch = np.asarray(np.max(output_batch, axis=3))
        output_batch = np.asarray(np.argmax(output_batch, axis=3), dtype=np.uint8)
        output_iterator = []
        heatmap_iterator = []
        scoremap_iterator = []

        for i in range(output_batch.shape[0]):
            output_iterator.append(output_batch[i,:,:])
            heatmap_iterator.append(heatmap_batch[i,:,:]/np.max(heatmap_batch[i,:,:]))
            scoremap_iterator.append(1-scoremap_batch[i,:,:]/np.max(scoremap_batch[i,:,:]))
            name_tmp = name[i].split('/')[-1]
            name[i] = '%s/%s' % (args.save, name_tmp)
        with Pool(4) as p:
            p.map(save, zip(output_iterator, name) )
            p.map(save_heatmap, zip(heatmap_iterator, name) )
            p.map(save_scoremap, zip(scoremap_iterator, name) )

        del output_batch

       

       

    
    return args.save

if __name__ == '__main__':
    tt = time.time()
    with torch.no_grad():
        save_path = main()
    print('Time used: {} sec'.format(time.time()-tt))
    os.system('python compute_iou.py ./data/Cityscapes/data/gtFine/val %s'%save_path)
