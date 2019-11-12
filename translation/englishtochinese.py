#!/usr/bin/python
# -*- coding: UTF-8 -*-

# import requests, json
#
# url = 'http://httpbin.org/post'
# data = {'key1': 'value1', 'key2': 'value2'}
# r = requests.post(url, data)
# print(r)
# print(r.text)
# print(r.content)
import random
import time

import requests
import numpy as np
import urllib.parse

url = 'http://translate.google.cn/translate_a/single?client=gtx&dt=t&ie=UTF-8&oe=UTF-8&sl=auto&tl=zh-CN&q='
fw=open("D:\\PyCharm 2019.2\\projects\\lstm_softmax\\translation\\out.txt","w",encoding='utf-8')
fopen = open('D:/PyCharm 2019.2/projects/lstm_softmax/dataset/dialog.txt', 'r', encoding='utf-8')
for line in fopen:
    if(line == "\n"):
        fw.write("\n")
    else:
        r = requests.get(url + line.replace("\n","").replace(" ", "%20"))
        arr = r.text.split("\"")
        fw.write((arr[1] + "\n"))
        fw.flush()
        time.sleep(random.uniform(0, 1))
fw.close()
fopen.close()



# # 先进行gb2312编码
# text = text.encode('gb2312')
# # 输出 b'\xd6\xd0\xce\xc4'
# # 再进行urlencode编码
# text = urllib.parse.quote(text)

# r = requests.get(url + text.replace(" ","%20"))
# arr = r.text.split("\"")
#
# print(arr[3])
# print(arr[1])
