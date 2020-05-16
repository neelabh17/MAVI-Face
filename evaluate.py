from __future__ import print_function
import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from data import cfg_mnet, cfg_re50
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
import cv2
from models.retinaface import RetinaFace
from utils.box_utils import decode, decode_landm
from utils.timer import Timer
from utils.evalResults import readData
from toolbox.label_pickle import makeLabelPickle
from toolbox.pickleOpers import save
from toolbox.makedir import make
import pickle
import cv2
from os.path import join
from widerface_evaluate.pipelineMAP import MAPCalcAfterEval
from toolbox.plotter import mapGraphPlotter

parser = argparser = argparse.ArgumentParser(description='Retinaface')
parser.add_argument('-m', '--trained_model', default='Resnet50_Final',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--network', default='resnet50', help='Backbone network mobile0.25 or resnet50')
parser.add_argument('--origin_size', default=True, type=str, help='Whether use origin image size to evaluate')
parser.add_argument('--save_folder', default='./widerface_evaluate/widerface_txt', type=str, help='Dir to save txt results')
parser.add_argument('--savename', default='baseline', type=str, help='name of our save')
parser.add_argument('--dataset', default='val', type=str, help="on which dataset do we compare images")
parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
parser.add_argument('--dataset_folder', default='./data/widerface/val/images', type=str, help='dataset path')
parser.add_argument('--confidence_threshold_eval', default=0.03, type=float, help='confidence_threshold_eval')
parser.add_argument('--confidence_threshold_infer', default=0.055, type=float, help='confidence_threshold_infer')
parser.add_argument('--area_threshold', default=225, type=float, help='area threshold to remove small faces')
parser.add_argument('--top_k', default=5000, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
parser.add_argument('-s', '--save_image', default="False",type=str, help='show detection results')
parser.add_argument('--merge_images', default="False",type=str, help='show detection results')
parser.add_argument('--vis_thres', default=0.5, type=float, help='visualization_threshold')
parser.add_argument('--mode', default='single', type=str, help='single: eval on single model, series: evaluate on training session comes with in built graph plotter')
args = parser.parse_args()


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


def eval(pretrained_path, mode=args.mode,seriesName=None,epoch=None):
    print("\nPerforming Evaluation for {}".format(pretrained_path))
    torch.set_grad_enabled(False)
    cfg = None
    if args.network == "mobile0.25":
        cfg = cfg_mnet
    elif args.network == "resnet50":
        cfg = cfg_re50
    # net and model
    net = RetinaFace(cfg=cfg, phase = 'test')
    net = load_model(net, pretrained_path, args.cpu)
    net.eval()
    print('Finished loading model!')
    print(net)
    cudnn.benchmark = True
    device = torch.device("cpu" if args.cpu else "cuda")
    net = net.to(device)

    # testset_folder = basically this is "./data/widerface/val/images/"
    testset_folder= args.dataset_folder
    print(testset_folder)
    # generating a pickle file for the labels
    labelLoc=join(os.path.dirname(testset_folder),"label.txt")
    makeLabelPickle(labelLoc)

    #importing gt data from pickle file
    neelTestDataset=readData(join(os.path.dirname(testset_folder), "label.pickle"))
    num_images = len(neelTestDataset)

    _t = {'forward_pass': Timer(), 'misc': Timer()}
    
    testResults={}
    # testing begin
    for i, img_name in enumerate(neelTestDataset):

        #unpacking dir nase
        directory,base=os.path.split(img_name)
        image_path = join(testset_folder, directory,base)
        print("image path is :",image_path)
        img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
        img = np.float32(img_raw)

        # testing scale
        target_size = 1600
        max_size = 2150
        im_shape = img.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        resize = float(target_size) / float(im_size_min)
        # prevent bigger axis from being more than max_size:
        if np.round(resize * im_size_max) > max_size:
            resize = float(max_size) / float(im_size_max)
        if args.origin_size:
            resize = 1

        if resize != 1:
            img = cv2.resize(img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
        im_height, im_width, _ = img.shape
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(device)
        scale = scale.to(device)

        _t['forward_pass'].tic()
        loc, conf, landms = net(img)  # forward pass
        _t['forward_pass'].toc()
        _t['misc'].tic()
        priorbox = PriorBox(cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
        scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                            img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                            img.shape[3], img.shape[2]])
        scale1 = scale1.to(device)
        landms = landms * scale1 / resize
        landms = landms.cpu().numpy()

        # print(boxes.shape,landms.shape,scores.shape)
        # helps in significant reduction of saving space
        inds = np.where(scores > args.confidence_threshold_eval)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]
            
        ##here boxes are giving us the real values x1,y1,x2,y2 we have to converet them into x1,y1,w,h
        boxes[..., 2]-=boxes[..., 0]
        boxes[..., 3]-=boxes[..., 1]
        
        imgResultDict={"conf":scores,"landms":landms,"loc":boxes}
        # adding it to the main results pickle file

        testResults[img_name]=imgResultDict        
        _t['misc'].toc()
        print('im_detect: {:d}/{:d} forward_pass_time: {:.4f}s misc: {:.4f}s'.format(i + 1, num_images, _t['forward_pass'].average_time, _t['misc'].average_time))
        
        #now saving the pickle file
        #saving in evalData folder
        if(mode=="single"):
            modelEvalFolder = join(os.getcwd(),"evalData",os.path.basename(pretrained_path).strip(".pth"),"outResults")
            save_name = join(modelEvalFolder,"outResults_{}.pickle".format(args.dataset))
        elif(mode=="series"):
            modelEvalFolder = join(os.getcwd(),"evalData",seriesName,"outResults")
            save_name = join(modelEvalFolder,"outResults_{}_epoch_{}.pickle".format(args.dataset,epoch))

        
        #make dir
        make(os.path.dirname(save_name))
        
        #save file
        save(testResults,save_name)

    print("Evaluation is done and evaluation file is saves as outResults.pickle in the respective folder of the model")
        #now calling evaluation simultaneously
        # os.chdir("./widerface_evaluate/")
        # str="./widerface_txt/"+args.pretrained_path.strip(".pth").strip("/weights/")+"/"
        # os.system("python neelPipelineEvaluation2.py --pred \'{}\'".format(str))
        
if __name__ == '__main__':
    if args.mode=="single":
        modelFile=args.trained_model+".pth"
        modelPath=join(os.getcwd(),"weights",modelFile)
        eval(modelPath, args.mode)
        MAPCalcAfterEval(args,modelPath)
    elif args.mode=="series":
        seriesName=args.trained_model
        
        # get list of all models trained in that series
        dirPath=join(os.getcwd(),"weights")
        modelList=os.listdir(dirPath)
        modelsInSeries=[]
        for model in modelList:
            if(seriesName in model):
                epochNumber=int(model.strip(".pth").split("_epoch_")[1])
                modelsInSeries.append([epochNumber,model])

        # my required models are stacked up
        #sorting them in order of epoch
        modelsInSeries.sort(key=lambda x:x[0])
        mapData=[]
        for myModelInfo in modelsInSeries:
            epochNo, modelFile=myModelInfo
            modelPath=join(os.getcwd(),"weights",modelFile)

            # evaluate results
            eval(modelPath, args.mode,seriesName=seriesName,epoch=epochNo)
            
            # get map
            MAP=MAPCalcAfterEval(args,modelPath,seriesData={"seriesName":seriesName,"epoch": epochNo})
            mapData.append({"epoch":epochNo,"map":MAP})
            mapDataSaverfile=join(os.getcwd(),"evalData",seriesName,"mapData","mapVsEpoch.pickle")
            
            # create directory
            make(os.path.dirname(mapDataSaverfile))

            #save data
            save(mapData,mapDataSaverfile)
            
            # plot graph
            mapGraphPlotter(mapDataSaverfile)

    else:
        print("Select correct mode")

        


