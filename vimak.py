
    # 'newOneToOneOhem_lr-beg=1.3e-03_lr-sch=1e-03_shuffle=False',
    # 'newOneToOneOhem_lr-beg=1.3e-03_lr-sch=None_shuffle=False',
from torch.utils.tensorboard import SummaryWriter
from toolbox.pickleOpers import loadup
ls=['5XOhemWithShuffleNoScheduler_epoch_36',
    'new_1xOhem_shuffle_true_scheduler_e2_epoch_22',
    'new_5xOhem_shuffle_true_scheduler_e2_epoch_10',
    # 'newOneToOneOhem_lr-beg=1.3e-03_lr-sch=1e-03_shuffle=False_epoch_40',
    # 'newOneToOneOhem_lr-beg=1.3e-03_lr-sch=None_shuffle=False_epoch_22',
    '1XOhemWithShuffleNoScheduler_lr-beg=1.3e-04_lr-sch=None_shuffle=True_epoch_40',
    'newOhemTrainSingleSampling32_lr-beg=1.3e-03_lr-sch=None_shuffle=True_epoch_1',
    'newOhemTrainSingleSampling32_lr-beg=1.3e-04_lr-sch=None_shuffle=True_epoch_4']
    # 'SingleSamplingOhemAdamLRe3_epoch_32',
    # 'Resnet50_Final']
# ls=['5XOhemWithShuffleNoScheduler',
#     'new_1xOhem_shuffle_true_scheduler_e2',
#     'new_5xOhem_shuffle_true_scheduler_e2',
#     '1XOhemWithShuffleNoScheduler_lr-beg=1.3e-04_lr-sch=None_shuffle=True',
#     'newOhemTrainSingleSampling32_lr-beg=1.3e-03_lr-sch=None_shuffle=True',
#     'newOhemTrainSingleSampling32_lr-beg=1.3e-04_lr-sch=None_shuffle=True','Renet50_Final','SingleSamplingOhemAdamLRe3_epoch_32']
writer=SummaryWriter("prData7")
for l in ls:
    a=loadup("evalData/{}_inferConf=0.7/prData/prCurve_val_epoch_{}.pickle".format(l.split("_epoch_")[0],l.split("_epoch_")[1]))
    
    for i in range(len(a)):
        writer.add_scalars("curve/",{l.split("_epoch_")[0]:a[i][0]},a[i][1]*1000)

ls=['SingleSamplingOhemAdamLRe3_epoch_32',
    'Resnet50_Final']
for l in ls:
    a=loadup("evalData/{}_inferConf=0.7/prData/prCurve_val.pickle".format(l))
    
    for i in range(len(a)):
        writer.add_scalars("curve/",{l:a[i][0]},a[i][1]*1000)


writer.close()

# # # ls=[ 'newOhemTrainSingleSampling32_lr-beg=1.3e-03_lr-sch=None_shuffle=True_epoch_1', 'newOhemTrainSingleSampling32_lr-beg=1.3e-04_lr-sch=None_shuffle=True_epoch_4']
# ls=[ 'newOneToOneOhem_lr-beg=1.3e-03_lr-sch=1e-03_shuffle=False_epoch_40', 'newOneToOneOhem_lr-beg=1.3e-03_lr-sch=None_shuffle=False_epoch_22']
# import os

# # os.system(f'python evaluate.py --trained_model "SingleSamplingOhemAdamLRe3_epoch_32" --confidence_threshold_infer 0.7')
# # os.system(f'python evaluate.py --trained_model "Resnet50_Final" --confidence_threshold_infer 0.7')
# for l in ls:
#     print(l)
#     os.system(f'python evaluate.py --trained_model "{l.split("_epoch_")[0]}" --mode "series"  --confidence_threshold_infer 0.7')



# for l in ls:
#     print(l)
#     # os.system(f'python detect.py --trained_model "{l}" --save_name "{l}" --vis_thres 0.055 --mode "images"')
#     os.system(f'python detect.py --trained_model "{l}" --save_name "{l}" --vis_thres 0.7 --mode "images"')
#     # os.system(f'python detect.py --trained_model "{l}" --save_name "{l}" --vis_thres 0.43 --mode "images"')
#     # os.system(f'python detect.py --trained_model "{l}" --save_name "{l}" --vis_thres 0.055 --mode "videos" --fps 1')
#     # os.system(f'python detect.py --trained_model "{l}" --save_name "{l}" --vis_thres 0.055 --mode "videos" --fps 5')
#     # os.system(f'python detect.py --trained_model "{l}" --save_name "{l}" --vis_thres 0.43 --mode "videos" --fps 5')
#     os.system(f'python detect.py --trained_model "{l}" --save_name "{l}" --vis_thres 0.7 --mode "videos" --fps 5')
#     os.system(f'python detect.py --trained_model "{l}" --save_name "{l}" --vis_thres 0.7 --mode "videos" --fps 1')
#     # os.system(f'python detect.py --trained_model "{l}" --save_name "{l}" --vis_thres 0.43 --mode "videos" --fps 1')
