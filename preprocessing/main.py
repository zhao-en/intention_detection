#!/usr/bin/python
# -*- coding: UTF-8 -*-


def delRepeat():
    dic = []
    fopenr = open('D:/PyCharm 2019.2/projects/lstm_softmax/dataset/910欢迎感谢再见', 'r', encoding='utf-8')
    fopenw = open('D:/PyCharm 2019.2/projects/lstm_softmax/dataset/910欢迎感谢再见_2', 'w', encoding='utf-8')
    for line in fopenr:
        if not line in dic:
            dic.append(line)
            fopenw.write(line)
            fopenw.flush()
    fopenw.close()
    fopenr.close()

delRepeat()