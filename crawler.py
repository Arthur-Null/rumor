#  -*- coding: utf-8 -*-
import time
import re
import tweepy
from tweepy import OAuthHandler


def readfile(file):     #read file
    f = open(file,"r")
    file_list = f.readlines()
    return file_list


def datetime_timestamp(dt):
    # dt为字符串
    # 中间过程，一般都需要将字符串转化为时间数组
    time.strptime(dt, '%Y-%m-%d %H:%M:%S')
    ## time.struct_time(tm_year=2012, tm_mon=3, tm_mday=28, tm_hour=6, tm_min=53, tm_sec=40, tm_wday=2, tm_yday=88, tm_isdst=-1)
    # 将"2012-03-28 06:53:40"转化为时间戳
    s = time.mktime(time.strptime(dt, '%Y-%m-%d %H:%M:%S'))
    return int(s)


consumer_key = '5WvQD5l4HeQMj0Ztonh9deVAb'
consumer_secret = '09fjeNOInbUehnwL3cmNruaJHaf22SzhR7iB9SjsBVqDeIyKEs'
access_token = '928893195705970688-0owztLK6IAEvEpmxPiQIlCmZyUKEU2E'
access_secret = '1nK6RVt9sfZUnOH11Hz2mieUI0F9u2eCTqskRxBHiMI1S'

auth = OAuthHandler(consumer_key,consumer_secret)
auth.set_access_token(access_token,access_secret)

api = tweepy.API(auth)

num = 0
id_list = readfile("Twitter.txt")
for i in id_list:
    tmp = re.split(r'[\s]',i)
    id = tmp[0]
    if tmp[1] == 'label:0':
        flag = 0
    else:
        flag = 1

    list = []
    output = []
    tmp1 = []

    cost = 0
    tweet_num = 2
    while(tweet_num < len(tmp)):
        # if num<= 499:    #从第几个event开始爬
        #     break
        tmp1.append(tmp[tweet_num])
        cost += 1
        tweet_num += 1
        if cost >= 99 or tweet_num >= len(tmp):
            cost = 0
            list = api.statuses_lookup(tmp1)
            for tweet in list:
                s = str(tweet.text)
                s = s.replace('\r', ' ')
                s = s.replace('\n', ' ')
                if s not in ['\t', '\r', ""] and str(tweet.created_at) not in ['\t', '\r', ""]:
                    add_str = s + "\t" + str(datetime_timestamp(str(tweet.created_at)))
                    # print(s,str(tweet.created_at))
                    # print(add_str)
                    output.append(add_str)
            tmp1 = []

    # if len(tmp) > 102:
    #     maxline = 102
    # else:
    #     maxline = len(tmp)
    # for j in range(2,maxline):
    #     tmp1.append(tmp[j])
    #
    # list = api.statuses_lookup(tmp1)
    #
    # for tweet in list:
    #     add_str = str(tweet.text) +"\t"+ str(datetime_timestamp(str(tweet.created_at)))
    #     output.append(add_str)
    num += 1
    print(num)

    # if num <= 499:   #从第几个event开始
    #     continue

    str_list = [line + '\n' for line in output]
    print(len(str_list))
    f = open(str("dataset/twitter/"+str(num)+"_"+str(flag)), 'w+', newline='', encoding='utf-8')
    f.writelines(str_list)
