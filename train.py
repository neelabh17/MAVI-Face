from __future__ import print_function
import os
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import argparse
import torch.utils.data as data
from data import WiderFaceDetection, detection_collate, preproc, cfg_mnet, cfg_re50,ohemDataSampler
from layers.modules import MultiBoxLoss
from layers.functions.prior_box import PriorBox
import time
import datetime
import math
from models.retinaface import RetinaFace
import pickle
from toolbox.plotter import lossGraphPlotter
from toolbox.makedir import make
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser(description='Retinaface Training')
parser.add_argument('--training_dataset', default='./data/widerface/train/label.txt', help='Training dataset directory')
parser.add_argument('--network', default='resnet50', help='Backbone network mobile0.25 or resnet50')
parser.add_argument('--num_workers', default=8, type=int, help='Number of workers used in dataloading')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--resume_net', default='./weights/Resnet50_Final.pth', help='resume net for retraining')
parser.add_argument('--resume_epoch', default=0, type=int, help='resume iter for retraining')
parser.add_argument('--save_epoch', default=2, type=int, help='after how many epoche steps should the model be saved')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
parser.add_argument('--save_folder', default='./weights/', help='Location to save checkpoint models')
parser.add_argument('--shuffle', default='False', help='Location to save checkpoint models')
# parser.add_argument('--lr_scheduler', default='False', help='Location to save checkpoint models')
parser.add_argument('--lr_scheduler_epsilon', default=1e-3, type=float, help='Weight decay for SGD')

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
if(args.shuffle=="True"):
    toShuffle=True
else:
    toShuffle=False
num_workers = args.num_workers
momentum = args.momentum
weight_decay = args.weight_decay
# initial_lr = args.lr
initial_lr = args.lr*batch_size/24
gamma = args.gamma
training_dataset = args.training_dataset
validation_dataset = args.validation_dataset
save_folder = args.save_folder
save_epoch=args.save_epoch

ohem_dataset = './data/widerface/ohem/label.txt'

net = RetinaFace(cfg=cfg)

# Updating model params before it is loaded
# net.BboxHead = net._make_bbox_head(fpn_num=5, inchannels=cfg['out_channel']) 
# net.ClassHead = net._make_class_head(fpn_num=5, inchannels=cfg['out_channel'])  
# import pdb;pdb.set_trace()

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
    # print(params)
    params.requires_grad = False
for params in net.ClassHead.parameters():  # set all layers requires_grad to false
    # print(params)
    params.requires_grad = True
for params in net.BboxHead.parameters():  # set all layers requires_grad to false
    # print(params)
    params.requires_grad = True

# re initialising our layers
# net.ClassHead = net._make_class_head(fpn_num=5, inchannels=cfg['out_channel'])  
# # we can think of redcing this fpn from 5 to 3 to increase inference time by a bit
# net.BboxHead = net._make_bbox_head(fpn_num=5, inchannels=cfg['out_channel']) 


Plist = []

for params in net.parameters():  # stores parameters that will be updated in Plist
    if params.requires_grad:
        Plist.append(params)


if num_gpu > 1 and gpu_train:  # now transfer net to gpu if possible
    net = torch.nn.DataParallel(net).cuda()
else:
    net = net.cuda()

# TODO change weight decay
# optimizer = optim.SGD(net.parameters(), lr=initial_lr, momentum=momentum, weight_decay=weight_decay)
optimizer = optim.Adam(Plist, lr=initial_lr, weight_decay=weight_decay)
criterion = MultiBoxLoss(num_classes, 0.35, True, 0, True, 7, 0.35, False)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    patience=3,
    factor=.3,
    threshold=args.lr_scheduler_epsilon,
    verbose=True)
# print(net)

priorbox = PriorBox(cfg, image_size=(img_dim, img_dim))
with torch.no_grad():
    priors = priorbox.forward()
    priors = priors.cuda()
    
def train():
    net.train()
    epoch = 0 + args.resume_epoch
    trainingSessionName=input("Enter the name for this training session: ")
    
    # removing any extra spaces from the input
    trainingSessionName=trainingSessionName.strip()
    trainingSessionName=f'{trainingSessionName}_lr-beg={initial_lr:.1e}_lr-sch={args.lr_scheduler_epsilon:.0e}_shuffle={toShuffle}'
    traingDetails=input("Enter details for the training : ")

    pwd=os.getcwd()
    intermediatePath=os.path.join("logs",trainingSessionName)
    sessionPath=os.path.join(pwd,intermediatePath)
    if(not os.path.isdir(sessionPath)):
        os.makedirs(sessionPath)
    
    f=open(os.path.join(sessionPath,"details.txt"),"w")
    f.write(traingDetails)
    f.close()

    print('Loading Train Dataset...')
    train_dataset = ohemDataSampler( training_dataset,preproc(img_dim, rgb_mean))
    # train_dataset = WiderFaceDetection( training_dataset,preproc(img_dim, rgb_mean))
    train_dataset_ = data.DataLoader(train_dataset,batch_size, shuffle=toShuffle, num_workers=num_workers, collate_fn=detection_collate)

    print('Loading Val Dataset...')
    # val_data = ohemDataSampler(validation_dataset,preproc(img_dim, rgb_mean))
    val_data = WiderFaceDetection(validation_dataset,preproc(img_dim, rgb_mean))
    dataset_ = data.DataLoader(val_data,batch_size, shuffle=toShuffle, num_workers=num_workers, collate_fn=detection_collate)

    print('Loading OHEM data...')
    # ohem_data = ohemDataSampler(ohem_dataset,preproc(img_dim, rgb_mean))
    ohem_data = WiderFaceDetection(ohem_dataset,preproc(img_dim, rgb_mean))
    ohem_data_ = data.DataLoader(ohem_data,batch_size, shuffle=toShuffle, num_workers=num_workers, collate_fn=detection_collate)

    epoch_size = math.ceil(len(train_dataset) / batch_size)
    max_iter = max_epoch * epoch_size

    stepvalues = (cfg['decay1'] * epoch_size, cfg['decay2'] * epoch_size)
    step_index = 0

    if args.resume_epoch > 0:
        start_iter = args.resume_epoch * epoch_size
    else:
        start_iter = 0

    epoch_loss_train = 0.0
    lossCollector=[]

    print("Setting up tensorboard")
    writer=SummaryWriter("trainLogs/{}".format(trainingSessionName),flush_secs=120)
   
    for iteration in range(start_iter, max_iter):
        if iteration % epoch_size == 0:
            # code called for each epoch at begin of the epoch
            # create batch iterator
            batch_iterator = iter(data.DataLoader(train_dataset, batch_size, shuffle=toShuffle, num_workers=num_workers, collate_fn=detection_collate))
            # for base model
            newtic=time.time()
            print("Performing Evalaution on the dataset at epoch {}".format(epoch))
            trainLoss=train_eval(net,train_dataset_,batch_size,epoch,mode=0)
            # trainLoss=epoch_loss_train
            valLoss=train_eval(net,dataset_,batch_size,epoch,mode=1)
            ohemLoss = train_eval(net,ohem_data_,batch_size,epoch,mode=2)
            lossCollector.append({"epoch":epoch,"trainLoss":trainLoss,"valLoss":valLoss,"ohemLoss":ohemLoss})
            # tensorboard logging
            writer.add_scalars("Loss per Epoch",
                                {"Train":trainLoss,
                                "Validation Loss": valLoss,
                                "Ohem Loss": ohemLoss},epoch)
            
            if (epoch % save_epoch == 0 and epoch > 0) :
                # code doest run for the zeroth epoch
                torch.save(net.state_dict(), save_folder + trainingSessionName+"_epoch_{}.pth".format(int(epoch)))
            
            scheduler.step(trainLoss)
            lr = optimizer.param_groups[0]['lr']
            writer.add_scalar("Learning Rate",lr,epoch)
            print("Done in {} secs".format(time.time()-newtic))
            
            #saving the losses data per epoch
            lossFolder=os.path.join(sessionPath,"lossData")
            make(lossFolder)
            lossDataFileName=os.path.join(lossFolder,"lossVsEpoch.pickle")
            picklefile=open(lossDataFileName,"wb")
            pickle.dump(lossCollector,picklefile)
            picklefile.close()

            # plotting the graph
            lossGraphPlotter(lossDataFileName) 




        load_t0 = time.time()
        if iteration in stepvalues:
            step_index += 1
        # lr = adjust_learning_rate(optimizer, gamma, epoch, step_index, iteration, epoch_size)
        lr = optimizer.param_groups[0]['lr']
        
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
        
        # TODO add this training data to tensorboard
        if iteration % epoch_size == 0:
            print('\nTraining loss for Epoch simultaneous wala {} : {}'.format(epoch,epoch_loss_train))
            writer.add_scalar("Simultaneous Train Loss per Epoch",epoch_loss_train,epoch)
            # writer.flush()
            epoch_loss_train=0
            epoch+=1

    # TODO save last file correctly
    torch.save(net.state_dict(), save_folder + trainingSessionName+"_epoch_{}.pth".format(int(epoch)))
    # torch.save(net.state_dict(), save_folder + cfg['name'] + '_Finally_FT_Adam_WC1.pth')
    writer.close()
    # torch.save(net.state_dict(), save_folder + 'Final_Retinaface.pth')

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


def train_eval(model,val_data,batch_size_val,epoch_no,mode = 1,is_base_model=False):
    '''
    mode= 0-> for training loss
    mode= 1-> for validation loss
    mode= 2-> for ohem loss
    '''

    model.eval()
    loss_val = 0.0
    i=0
    totImg=0

    for images_,targets_ in val_data:
        totImg+=(images_.shape[0])
        print("{} done out of {}".format(i,len(val_data)))
        i+=1
        
        images_ = images_.cuda()  # send to gpu
        targets_ = [anno.cuda() for anno in targets_]

        with torch.no_grad():
            out = model(images_)
            loss_l, loss_c, loss_landm = criterion(out, priors, targets_)
            loss = cfg['loc_weight'] * loss_l + loss_c + loss_landm
            loss_val += loss.item()
    
    loss_val = loss_val /(totImg)  # get average loss per image
    if(mode==0):
        # if running evaluation on training set for the pretrained model
        print('Training loss for Epoch {} : {}'.format(epoch_no,loss_val))
    elif(mode==1):
        print('Validation loss for Epoch {} : {}'.format(epoch_no,loss_val))
    elif(mode==2):
        print('Ohem loss for Epoch {} : {}'.format(epoch_no,loss_val))
    return loss_val
    # print('Validation loss per image for Epoch {} : {}'.format(epoch_no,loss_val))


if __name__ == "__main__":
    train()
