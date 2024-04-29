import glob
import os
import cv2
import numpy as np

src_dir = "c:/proj/ComfyUI/script/inc"
out_dir = "c:/proj/ComfyUI/script/inc/out"

if not os.path.exists(out_dir):
    os.mkdir(out_dir)

files = glob.glob(src_dir + "/*.png")
files = [file for file in files if "_" not in file]

for f in files:
    fn = os.path.splitext(f)[0]
    bn = os.path.basename(fn)
    
    og = cv2.imread(fn + ".png")
    gen = cv2.imread(fn + "_generated.png")

    og  = cv2.resize(og, (512, 512))

    res = np.hstack((og, gen))
    print(f"{out_dir}/{bn}.jpg")
    cv2.imwrite(f"{out_dir}/{bn}.jpg", res)
