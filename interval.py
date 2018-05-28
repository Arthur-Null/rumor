import sys
import numpy as np
import pickle as pkl
import os


def get_interval(tweet, time, N):
    assert len(tweet) == len(time)
    start = np.min(time)
    end = np.max(time)
    L = float(end - start)
    #print(time)
    l = L/N
    last = 0
    m = 0
    tail = 0
    while True:
        print(l)
        flag = [False] * int(((end-start)/l))
        for t in time:
            index = int((t-start)/l)
            if index == int(((end-start)/l)):
                index -= 1
            flag[index] = True
        m = 0
        tail = 0
        tmp = 0
        for i in range(len(flag)):
            if flag[i]:
                tmp += 1
            else:
                if(tmp > m):
                    m = tmp
                    tail = i
                tmp = 0
        if (tmp > m):
            m = tmp
            tail = len(flag)
        if m < N and m > last:
            l = l * 0.5
            last = m
        else:
            break
    result = [""] * m
    for i in range(len(time)):
        t = time[i]
        index = int((t-start)/l) - tail + m
        if i == len(time)-1:
            index -= 1
        if 0 <= index < m:
            result[index] += (tweet[i] + ' ')
    return result

def separate_file(path):
    fin = open('dataset/raw/' + path, 'r', encoding='utf8')
    tweet = []
    time = []
    lines = fin.readlines()
    if len(lines) <= 2:
        return
    fout = open('dataset/sep/' + path[:-4] + '.pkl', 'wb')
    for line in lines:
        try:
            time.append(int(line.split('\t')[-1]))
            tweet.append(line[:-11])
        except:
            break
    print(path)
    result = get_interval(tweet, time, 30)
    pkl.dump(result, fout)
    fout.close()
    fin.close()

for (root, dir, files) in os.walk("dataset/raw"):
    for f in files:
        separate_file(f)
