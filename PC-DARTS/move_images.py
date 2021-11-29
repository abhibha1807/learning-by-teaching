import shutil
import os
from os import listdir
from os.path import isfile, join
filename1 = '/abhibha-volume/PCDARTS-cifar10/data/train/NORMAL/'
filename2 = '/abhibha-volume/PCDARTS-cifar10/data/train/PNEUMONIA/'
filename5 = '/abhibha-volume/PCDARTS-cifar10/data_copy/train/NORMAL/'
filename6 = '/abhibha-volume/PCDARTS-cifar10/data_copy/train/PNEUMONIA/'
    
onlyfiles1 = [f for f in listdir(filename1) if isfile(join(filename1, f))]
onlyfiles2 = [f for f in listdir(filename2) if isfile(join(filename2, f))]
onlyfiles5 = [f for f in listdir(filename5) if isfile(join(filename5, f))]
onlyfiles6 = [f for f in listdir(filename6) if isfile(join(filename6, f))]

    
for i in onlyfiles1:
    towrite = './cleaned_train/NORMAL/'
    img_name = filename1 + i
    #plt.imshow(img_)
    l=img.shape[0]
    w=img.shape[1]
    
    img = img_.reshape(l,w,3)
    print(img.shape)
    cv2.imwrite(towrite+i, result)
    print('written:', towrite+i)
    print(c)

print('cleaned train NORMAL finished')
shutil.move("/abhibha-volume/Image-Enhancer/cleaned_train_copy/", "/abhibha-volume/PCDARTS-cifar10/cleaned_train/NORMAL/"+)