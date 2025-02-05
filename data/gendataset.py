# com

import os
from tqdm import tqdm
import shutil
import cv2
import threading
import time

sourcepath = 'E:/CCPD/download/CCPD/raw/CCPD2019/ccpd_base'
labelPath = 'E:/CCPD/final/labels/base'
picPath = 'E:/CCPD/final/images/base'
files=os.listdir(sourcepath)
files.sort()

def mov(num:int):
    if not os.path.exists(labelPath+str(num)):
        os.mkdir(labelPath+str(num))
    if not os.path.exists(picPath+str(num)):
        os.mkdir(picPath+str(num))
    print(num*5000-5000,num*5000)
    for filename in tqdm(files[num*5000-5000:num*5000]):
        shutil.copyfile(os.path.join(sourcepath,filename), os.path.join(picPath+str(num),filename))

        list1 = filename.split("-", 3)  # 第一次分割，以减号'-'做分割
        subname = list1[2]
        lt, rb = subname.split("_", 1)  # 第二次分割，以下划线'_'做分割
        lx, ly = lt.split("&", 1)
        rx, ry = rb.split("&", 1)
        width = int(rx) - int(lx)
        height = int(ry) - int(ly)  # bounding box的宽和高
        cx = float(lx) + width / 2
        cy = float(ly) + height / 2  # bounding box中心点
        # print("#2,",filename)
        img = cv2.imread(os.path.join(sourcepath,filename))
        width = width / img.shape[1]
        height = height / img.shape[0]
        cx = cx / img.shape[1]
        cy = cy / img.shape[0]

        txtname = filename.split(".", 1)
        txtfile = os.path.join(labelPath+str(num),txtname[0] + ".txt")
        # 绿牌是第0类，蓝牌是第1类
        with open(txtfile, "w") as f:  # w表示写入txt文件，如果txt文件不存在将会创建
            f.write(str(0) + " " + str(cx) + " " + str(cy) + " " + str(width) + " " + str(height))


if __name__ == "__main__":
    

    for i in range(20,40):
        print("Starting thread for %d" % i)
        try:
            thread = threading.Thread(target=mov,name='mov-%d'%i,args=(i,))
            thread.start()
        except:
            print ("Error: Unable to start thread for %d" % i)
    
    while threading.active_count()!=1:
    # threading.enumerate(): 返回一个包含正在运行的线程的list，包含线程名称和标识id。
    # thread_num = len(threading.enumerate())
        print("There are currently %d threads running." % threading.active_count())
    # print(threading.enumerate())
    # if thread_num <= 1:
    #     break
        time.sleep(1)