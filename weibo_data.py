#  -*- coding: utf-8 -*-
import time
import re
import os
import json
path = path = 'dataset/Weibo'

def readfile(file):     #read file
    f = open(file,"r")
    file_list = f.readlines()
    return file_list

num = 0
id_list = readfile("dataset/Weibo.txt")
for i in id_list:
    tmp = re.split(r'[\s]',i)
    id = tmp[0]
    if tmp[1] == 'label:0':
        flag = 0
    else:
        flag = 1
    list = []
    output = []
    # for j in range(2,maxline):
    for file in os.listdir(path):
        if file[:-5] == tmp[0][4:]:
            with open("dataset/Weibo/"+file, encoding='utf-8') as f1:
                d = json.load(f1)
                for k in range (len(d)):
                    if d[k]["text"] in ["转发微博。" , "转发微博" , "轉發微博" , "轉發微博。"]:
                        continue
                    else:
                        add_str = str(d[k]["text"]) + "\t" + str(d[k]["t"])
                        output.append(add_str)

    str_list = [line + '\n' for line in output]
    f = open(str("dataset/weibo_raw/"+str(num)+"_"+str(flag)), 'w+', newline='', encoding='utf-8')
    f.writelines(str_list)
    num += 1
    print(num)
