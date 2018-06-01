import os
import shutil
path = 'dataset/weibo_sep'         # 替换为你的路径\
traindir = 'dataset/weibo_train/'
testdir =  'dataset/weibo_test/'

dir = os.listdir(path)
one = 0
zero =0
for i in dir:
    if i[-5]=='1':
        one+=1
    else:
        zero+=1
print(one,zero)

train_one = 1900
train_zero = 1900

zero_num  = 0
one_num = 0
for file in os.listdir(path):
    srcFile = os.path.join(path,file)
    if file[-5] == '1':
        if one_num < train_one:
            targetFile = os.path.join(traindir,file)
            shutil.copyfile(srcFile,targetFile)
            one_num+=1
        else:
            targetFile = os.path.join(testdir, file)
            shutil.copyfile(srcFile, targetFile)
    else:
        if zero_num < train_zero:
            targetFile = os.path.join(traindir, file)
            shutil.copyfile(srcFile, targetFile)
            zero_num+=1
        else:
            targetFile = os.path.join(testdir, file)
            shutil.copyfile(srcFile, targetFile)