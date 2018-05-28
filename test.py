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

num = 1
list = readfile("Twitter.txt")
for i in list:
    tmp = re.split(r'[\s]',i)
    id = tmp[0]
    if tmp[1] == 'label:0':
        flag = 0
    else:
        flag = 1
    list = []
    output = []
    tmp1 = []
    if len(tmp) > 102:
        maxline = 102
    else:
        maxline = len(tmp)
    for j in range(2,maxline):
        tmp1.append(tmp[j])

    list = api.statuses_lookup(tmp1)

    for tweet in list:
        add_str = str(tweet.text).replace('\n', ' ').replace('\r', ' ') +"\t"+ str(datetime_timestamp(str(tweet.created_at)))
        output.append(add_str)
    str_list = [line + '\n' for line in output]
    f = open(str("dataset/raw/"+str(num)+"_"+str(flag)), 'w+', newline='', encoding='utf-8')
    f.writelines(str_list)

    num += 1
    print(num)

    # for tweet in range(2,len(tmp)):
    #     print(num,tmp[tweet])
    #     word = api.get_status(tmp[tweet])
    #     list.append(word)
    #     num += 1
    #     print(word.text,word.created_at)
    #     add_str = str(word.text) + str(word.created_at)
    #     f = open("1.txt", 'a+', newline='', encoding='utf-8')
    #     f.writelines(add_str)
    #     f.close()

# print(api.get_status(671868848455335936))


# for status in tweepy.Cursor(api.get_status(1000337778691989510)):
#     print (status.text)