from __future__ import print_function
import os
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import argparse
import torch.utils.data as data
from data import WiderFaceDetection, detection_collate, preproc, cfg_mnet, cfg_re50
from layers.modules import MultiBoxLoss
from layers.functions.prior_box import PriorBox
import time
import datetime
import math
from models.retinaface import RetinaFace
import pickle

parser = argparse.ArgumentParser(description='Retinaface Training')
parser.add_argument('--training_dataset', default='./data/widerface/train/label.txt', help='Training dataset directory')
parser.add_argument('--network', default='resnet50', help='Backbone network mobile0.25 or resnet50')
parser.add_argument('--num_workers', default=4, type=int, help='Number of workers used in dataloading')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--resume_net', default='./weights/Resnet50_Final.pth', help='resume net for retraining')
parser.add_argument('--resume_epoch', default=0, type=int, help='resume iter for retraining')
parser.add_argument('--save_epoch', default=2, type=int, help='after how many epoche steps should the model be saved')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
parser.add_argument('--save_folder', default='./weights/', help='Location to save checkpoint models')

parser.add_argument('--validation_dataset', default='./data/widerface/val/label.txt', help='Validation dataset directory')

args = parser.parse_args()

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)
cfg = None
if args.network == "mobile0.25":
    cfg = cfg_mnet
elif args.network == "resnet50":
    cfg = cfg_re50

rgb_mean = (104, 117, 123) # bgr order
num_classes = 2
img_dim = cfg['image_size']
num_gpu = cfg['ngpu']
batch_size = cfg['batch_size']
max_epoch = cfg['epoch']
gpu_train = cfg['gpu_train']

num_workers = args.num_workers
momentum = args.momentum
weight_decay = args.weight_decay
initial_lr = args.lr
gamma = args.gamma
training_dataset = args.training_dataset
validation_dataset = args.validation_dataset
save_folder = args.save_folder

ohem_dataset = './data/widerface/ohem/label.txt'

net = RetinaFace(cfg=cfg)

# resume net if possible
if args.resume_net is not None:
    print('Loading resume network...')
    state_dict = torch.load(args.resume_net)
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        head = k[:7]
        if head == 'module.':
            name = k[7:] # remove `module.`
        else:
            name = k
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict)

cudnn.benchmark = True

# okay now we want to re-initialise layers

for params in net.parameters():  # set all layers requires_grad to false
    params.requires_grad = False

# re initialising our layers
net.ClassHead = net._make_class_head(fpn_num=3, inchannels=cfg['out_channel'])  
# we can think of redcing this fpn from 5 to 3 to increase inference time by a bit
net.BboxHead = net._make_bbox_head(fpn_num=3, inchannels=cfg['out_channel']) 


Plist = []

for params in net.parameters():  # stores parameters that will be updated in Plist
    if params.requires_grad:
        Plist.append(params)

if num_gpu > 1 and gpu_train:  # now transfer net to gpu if possible
    net = torch.nn.DataParallel(net).cuda()
else:
    net = net.cuda()


# optimizer = optim.SGD(net.parameters(), lr=initial_lr, momentum=momentum, weight_decay=weight_decay)
optimizer = optim.Adam(Plist, lr=initial_lr, weight_decay=weight_decay)
criterion = MultiBoxLoss(num_classes, 0.35, True, 0, True, 7, 0.35, False)

# print(net)

priorbox = PriorBox(cfg, image_size=(img_dim, img_dim))
with torch.no_grad():
    priors = priorbox.forward()
    priors = priors.cuda()

def train():
    net.train()
    epoch = 0 + args.resume_epoch
    pickleFileSaverName=input("Enter the name to save loss data : ")
    print('Loading Dataset...')

    dataset = WiderFaceDetection( training_dataset,preproc(img_dim, rgb_mean))

    print('Loading Val Dataset...')
    val_data = WiderFaceDetection(validation_dataset,preproc(img_dim, rgb_mean))
    dataset_ = data.DataLoader(val_data,batch_size, shuffle=True, num_workers=num_workers, collate_fn=detection_collate)

    print('Loading OHEM data...')
    ohem_data = WiderFaceDetection(ohem_dataset,preproc(img_dim, rgb_mean))
    ohem_data_ = data.DataLoader(ohem_data,batch_size, shuffle=True, num_workers=num_workers, collate_fn=detection_collate)

    epoch_size = math.ceil(len(dataset) / batch_size)
    max_iter = max_epoch * epoch_size

    stepvalues = (cfg['decay1'] * epoch_size, cfg['decay2'] * epoch_size)
    step_index = 0

    if args.resume_epoch > 0:
        start_iter = args.resume_epoch * epoch_size
    else:
        start_iter = 0

    epoch_loss_train = 0.0
    lossCollector=[]

    for iteration in range(start_iter, max_iter):
        if iteration % epoch_size == 0:
            # code called for each epoch at begin of the epoch
            # create batch iterator
            batch_iterator = iter(data.DataLoader(dataset, batch_size, shuffle=True, num_workers=num_workers, collate_fn=detection_collate))
            if (epoch % 2 == 0 and epoch > 0) :
                torch.save(net.state_dict(), save_folder + cfg['name']+ '_epoch_' + str(epoch) + '_noGrad_FT_Adam_WC1.pth')
            epoch += 1

        load_t0 = time.time()
        if iteration in stepvalues:
            step_index += 1
        lr = adjust_learning_rate(optimizer, gamma, epoch, step_index, iteration, epoch_size)
        # load train data
        images, targets = next(batch_iterator)
        images = images.cuda()
        targets = [anno.cuda() for anno in targets]

        # forward
        out = net(images)

        # backprop
        optimizer.zero_grad()
        loss_l, loss_c, loss_landm = criterion(out, priors, targets)
        loss = cfg['loc_weight'] * loss_l + loss_c + loss_landm
        loss.backward()
        optimizer.step()
        load_t1 = time.time()
        batch_time = load_t1 - load_t0
        eta = int(batch_time * (max_iter - iteration))
        epoch_loss_train += loss.item()
        print('Epoch:{}/{} || Epochiter: {}/{} || Iter: {}/{} || Loc: {:.4f} Cla: {:.4f} Landm: {:.4f} || LR: {:.8f} || Batchtime: {:.4f} s || ETA: {}'
              .format(epoch, max_epoch, (iteration % epoch_size) + 1,
              epoch_size, iteration + 1, max_iter, loss_l.item(), loss_c.item(), loss_landm.item(), lr, batch_time, str(datetime.timedelta(seconds=eta))))

        # calculate validation loss after each epoch
        if iteration % epoch_size == 0:
            # run for each epoch at then end of each epoch

            # print('Training loss per image for Epoch {} : {}'.format(epoch,epoch_loss_train/len(dataset)))
            print('Training loss for Epoch {} : {}'.format(epoch,epoch_loss_train))

            # calc val loss after each epoch
            valLoss=train_eval(net,dataset_,batch_size,epoch)

            # also calc ohem_loss after each epoch
            ohem_loss = train_eval(net,ohem_data_,batch_size,epoch,is_ohem=True)

            lossCollector.append({"Epoch":epoch,"TrainLoss":epoch_loss_train,"ValLoss":valLoss,"OhemLoss":ohem_loss})
            
            #saving the losses data per epoch
            picklefile=open("./lossData/"+pickleFileSaverName+"_{}.pickle".format(epoch),"wb")
            pickle.dump(lossCollector,picklefile)
            picklefile.close()
            #saving is complete

            epoch_loss_train = 0.0

    torch.save(net.state_dict(), save_folder + cfg['name'] + '_Finally_FT_Adam_WC1.pth')
    # torch.save(net.state_dict(), save_folder + 'Final_Retinaface.pth')

    #saving the data for losses
    picklefile=open("./lossData/"+pickleFileSaverName+".pickle" ,"wb")
    pickle.dump(lossCollector,picklefile)

    picklefile.close()

def adjust_learning_rate(optimizer, gamma, epoch, step_index, iteration, epoch_size):
    """Sets the learning rate
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    warmup_epoch = -1
    if epoch <= warmup_epoch:
        #basically this isnt going to run like ever
        lr = 1e-6 + (initial_lr-1e-6) * iteration / (epoch_size * warmup_epoch)
    else:
        #we are going to run this one
        lr = initial_lr * (gamma ** (step_index))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def train_eval(model,val_data,batch_size_val,epoch_no,is_ohem = False):
    model.eval()
    loss_val = 0.0
    for images_,targets_ in val_data:
        
        images_ = images_.cuda()  # send to gpu
        targets_ = [anno.cuda() for anno in targets_]

        with torch.no_grad():
            out = model(images_)
            loss_l, loss_c, loss_landm = criterion(out, priors, targets_)
            loss = cfg['loc_weight'] * loss_l + loss_c + loss_landm
            loss_val += loss.item()
    if not is_ohem:
        # loss_val = loss_val / (batch_size * len(val_data))  # get average loss per image
        print('Validation loss for Epoch {} : {}'.format(epoch_no,loss_val))
    else:
        print('Ohem loss for Epoch {} : {}'.format(epoch_no,loss_val))
    return loss_val
    # print('Validation loss per image for Epoch {} : {}'.format(epoch_no,loss_val))


if __name__ == "__main__":
    train()