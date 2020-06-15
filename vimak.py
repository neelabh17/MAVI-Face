# ls=['5XOhemWithShuffleNoScheduler_epoch_36', 'new_1xOhem_shuffle_true_scheduler_e2_epoch_22', 'new_5xOhem_shuffle_true_scheduler_e2_epoch_10', 'newOneToOneOhem_lr-beg=1.3e-03_lr-sch=1e-03_shuffle=False_epoch_40', 'newOneToOneOhem_lr-beg=1.3e-03_lr-sch=None_shuffle=False_epoch_22', '1XOhemWithShuffleNoScheduler_lr-beg=1.3e-04_lr-sch=None_shuffle=True_epoch_40','SingleSamplingOhemAdamLRe3_epoch_32','Resnet50_Final']
ls=[ 'newOneToOneOhem_lr-beg=1.3e-03_lr-sch=1e-03_shuffle=False_epoch_40', 'newOneToOneOhem_lr-beg=1.3e-03_lr-sch=None_shuffle=False_epoch_22']
import os
for l in ls:
    print(ls)
    # os.system(f'python detect.py --trained_model "{l}" --save_name "{l}" --vis_thres 0.055 --mode "images"')
    os.system(f'python detect.py --trained_model "{l}" --save_name "{l}" --vis_thres 0.43 --mode "images"')
    # os.system(f'python detect.py --trained_model "{l}" --save_name "{l}" --vis_thres 0.055 --mode "videos" --fps 1')
    # os.system(f'python detect.py --trained_model "{l}" --save_name "{l}" --vis_thres 0.055 --mode "videos" --fps 5')
    os.system(f'python detect.py --trained_model "{l}" --save_name "{l}" --vis_thres 0.43 --mode "videos" --fps 5')
    os.system(f'python detect.py --trained_model "{l}" --save_name "{l}" --vis_thres 0.43 --mode "videos" --fps 1')