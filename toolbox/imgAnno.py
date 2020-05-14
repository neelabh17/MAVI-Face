import cv2
import matplotlib.pyplot as plt

def putbbox(filepath,dets):
    print(filepath)
    img_raw=cv2.imread(filepath)
    print(img_raw.shape)

    for b in dets:
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
    plt.clf()
    plt.imshow(img_raw[:,:,::-1])
    plt.show()

def test():
    import os
    from os.path import join
    from toolbox.pickleOpers import loadup

    a=loadup("data\widerface\ohem\label.pickle")
    for file in a:
        c,b=os.path.split(file)
        tp=join("data\widerface\ohem\images",c,b)
        putbbox(tp,a[file])
