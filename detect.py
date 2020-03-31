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
import time
import numpy as np

parser = argparse.ArgumentParser(description='Retinaface')

parser.add_argument('-m', '--trained_model', default='./weights/Resnet50_Final.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--network', default='resnet50', help='Backbone network mobile0.25 or resnet50')
parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
parser.add_argument('--confidence_threshold', default=0.5, type=float, help='confidence_threshold')
parser.add_argument('--top_k', default=5000, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
parser.add_argument('-s', '--save_image', action="store_true", default=True, help='show detection results')
parser.add_argument('--vis_thres', default=0.6, type=float, help='visualization_threshold')
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


if __name__ == '__main__':
    torch.set_grad_enabled(False)
    cfg = None
    if args.network == "mobile0.25":
        cfg = cfg_mnet
    elif args.network == "resnet50":
        cfg = cfg_re50
    # net and model
    net = RetinaFace(cfg=cfg, phase = 'test')
    net = load_model(net, args.trained_model, args.cpu)
    net.eval()
    print('Finished loading model!')
    print(net)
    cudnn.benchmark = True
    device = torch.device("cpu" if args.cpu else "cuda")
    net = net.to(device)

    resize = 1
    # ok=(0,0)
    # i=600
    # notFound=True
    # gfn=True
    # while(notFound and i<800):

    #     j=600
    #     if(not gfn):
    #         gfn=True
    #     while(notFound and j<800 and gfn):

    #         priorbox = PriorBox(cfg, image_size=(i, j))
    #         priors = priorbox.forward()
    #         priors = priors.to(device)
    #         prior_data = priors.data
    #         print(" This is the prior box shape",prior_data.shape[0],i,j)

    #         if(prior_data.shape[0]==16800):
    #             notFound=False
    #             ok=(i,j)    
    #         if(prior_data.shape[0]>16800):
    #             gfn=False
    #         j+=1
    #     i+=1

    # print("the ok value is the following :", ok)
    # testing begin
    for i in range(1):


        image_path = "nano.jpg"
        img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
        print("The original image shape is ", img_raw.shape )
        img_raw=cv2.resize(img_raw,(633,633))
        img = np.float32(img_raw)
        im_height, im_width, _ = img.shape
        print("The image shape is ", img.shape )
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(device)
        scale = scale.to(device)


        import pickle
        model_name="faceDetector"
        dbfile = open(model_name, 'rb')      
        db = pickle.load(dbfile) 
        loc2,conf2,landms2=db["conf"],db["loc"],db["landmarks"]
        loc2=torch.Tensor(loc2).to(device)
        conf2=torch.Tensor(conf2).to(device)
        landms2=torch.Tensor(landms2).to(device)
        ok=6500
        # loc2=loc2[:,ok:8064+ok,:]
        # conf2=conf2[:,ok:8064+ok,:]
        # landms2=landms2[:,ok:8064+ok,:]

        tic = time.time()
        loc, conf, landms = net(img)  # forward pass


        print(loc2.shape,conf2.shape,landms2.shape)

        print('net forward time: {:.4f}'.format(time.time() - tic))

        priorbox = PriorBox(cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(device)
        prior_data = priors.data
        print(" This is the prior box shape",prior_data.shape)

        boxes2 = decode(loc2.data.squeeze(0), prior_data, cfg['variance'])
        boxes2 = boxes2 * scale / resize
        boxes2 = boxes2.cpu().numpy()
        scores2 = conf2.squeeze(0).data.cpu().numpy()[:, 1]
        print(max(scores2))
        input()

        boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        print(max(scores))
        input()

        landms2 = decode_landm(landms2.data.squeeze(0), prior_data, cfg['variance'])
        landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])

        # print(landms)
        scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2]])
        scale1 = scale1.to(device)
        landms2 = landms2 * scale1 / resize
        landms2 = landms2.cpu().numpy()

        landms = landms * scale1 / resize
        landms = landms.cpu().numpy()

        # i=0
        # for b in arrlandms:
        #         if(i%10==0):
        #             cv2.circle(img_raw, (b[0], b[1]), 1, (0, 0, 255), 4)
        #             cv2.circle(img_raw, (b[2], b[3]), 1, (0, 255, 255), 4)
        #             cv2.circle(img_raw, (b[4], b[5]), 1, (255, 0, 255), 4)
        #             cv2.circle(img_raw, (b[6], b[7]), 1, (0, 255, 0), 4)
        #             cv2.circle(img_raw, (b[8], b[9]), 1, (255, 0, 0), 4)
        #         i+=1

        print(landms2.shape)
        # ignore low scores
        inds2 = np.where(scores2 > args.confidence_threshold)[0]
        boxes2 = boxes2[inds2]
        landms2 = landms2[inds2]
        scores2 = scores2[inds2]
        print(landms2.shape)

        # keep top-K before NMS
        order2 = scores2.argsort()[::-1][:args.top_k]
        boxes2 = boxes2[order2]
        landms2 = landms2[order2]
        scores2 = scores2[order2]
        print(landms2.shape)

        # do NMS
        dets2 = np.hstack((boxes2, scores2[:, np.newaxis])).astype(np.float32, copy=False)
        keep2 = py_cpu_nms(dets2, args.nms_threshold)
        print(landms2.shape)
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets2= dets2[keep2, :]
        landms2 = landms2[keep2]
        print(landms2.shape)

        # keep top-K faster NMS
        dets2 = dets2[:args.keep_top_k, :]
        landms2 = landms2[:args.keep_top_k, :]
        print(landms2.shape)

        dets2 = np.concatenate((dets2, landms2), axis=1)
        # print(dets2.shape)
        
        
        
        # ignore low scores
        inds = np.where(scores > args.confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:args.top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, args.nms_threshold)
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        dets = dets[:args.keep_top_k, :]
        landms = landms[:args.keep_top_k, :]

        dets = np.concatenate((dets, landms), axis=1)
        # img_raw=cv2.resize(img_raw,(383,512))
        # show image
        if args.save_image:
            for b in dets2:
                if b[4] < args.vis_thres:
                    continue
                text = "{:.4f}".format(b[4])
                b = list(map(int, b))
                cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
                cx = b[0]
                cy = b[1] + 12
                cv2.putText(img_raw, text, (cx, cy),
                            cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

                # landms
                cv2.circle(img_raw, (b[5], b[6]), 1, (0, 0, 255), 4)
                cv2.circle(img_raw, (b[7], b[8]), 1, (0, 255, 255), 4)
                cv2.circle(img_raw, (b[9], b[10]), 1, (255, 0, 255), 4)
                cv2.circle(img_raw, (b[11], b[12]), 1, (0, 255, 0), 4)
                cv2.circle(img_raw, (b[13], b[14]), 1, (255, 0, 0), 4)
            # save image

            name = "test.jpg"
            # anti transposing



            cv2.imwrite(name, img_raw)
        # name = "test.jpg"
        # cv2.imwrite(name, img_raw)

