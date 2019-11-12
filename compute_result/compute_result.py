#!/usr/bin/python
# -*- coding: UTF-8 -*-

# TP TN FP FN

main_word2vec_lstm_softmax =[[140, 4, 9, 0, 0], [12, 50, 0, 1, 3], [11, 8, 56, 1, 2], [1, 6, 1, 11, 0], [0, 2, 5, 0, 19]]
# W2V-GRU-Softmax = [[116, 9, 27, 1, 0], [7, 51, 6, 1, 1], [2, 5, 71, 0, 0], [1, 6, 1, 10, 1], [0, 1, 3, 0, 22]]

# W2V-LSTM-Softmax = [[140, 4, 9, 0, 0], [12, 50, 0, 1, 3], [11, 8, 56, 1, 2], [1, 6, 1, 11, 0], [0, 2, 5, 0, 19]]

# W2V-BiLSTM-Softmax = [[137, 6, 10, 0, 0], [16, 43, 2, 1, 4], [11, 4, 61, 2, 0], [2, 7, 0, 10, 0], [0, 5, 2, 0, 19]]


# TN
# TP = 0
# for index , X1 in enumerate(main_word2vec_lstm_softmax):
#     TP += main_word2vec_lstm_softmax[index][index]
#
# # FN A分到了非A
# FN = 0
# for index , X1 in enumerate(main_word2vec_lstm_softmax):
#     for index2 , x2 in enumerate(X1):
#         if index == index2:
#             continue
#         else:
#             FN = FN + int(x2)
#
# #FP 非A 分到了A
# FP = 0
# for index , X1 in enumerate(main_word2vec_lstm_softmax):
#     for index2 , x2 in enumerate(main_word2vec_lstm_softmax):
#         if index != index2:
#             FP = FP + main_word2vec_lstm_softmax[index2][index]
#
#
# # 非A 分到了 非A
# type = 0
# TN = 0
# for index, X1 in enumerate(main_word2vec_lstm_softmax):
#     for index_x ,x2 in enumerate(main_word2vec_lstm_softmax):
#         for index_y ,x2 in enumerate(main_word2vec_lstm_softmax):
#             if index != index_x and index != index_y:
#                 TN += main_word2vec_lstm_softmax[index_x][index_y]
#
# debug = 0
#
#
# Accuracy  = (TP + TN)/ (TP + TN + FP + FN)
#
# Precision = TP /(TP + FP)
#
# Recall  = TP /(TP + FN)
#
# f1 = 2 *Precision * Recall / (Precision + Recall)


import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

y_true = []
y_pred = []
for index_x in range(5):
    for index_y in range(5):
        for count in range(main_word2vec_lstm_softmax[index_x][index_y]):
            y_true.append(index_x)
            y_pred.append(index_y)


Accuracy = accuracy_score(y_true, y_pred)

Macro_Average = precision_score(y_true, y_pred, average='macro')
micro_Average = precision_score(y_true, y_pred, average='micro')

Macro_recall = recall_score(y_true, y_pred, average='macro')
micro_recall = recall_score(y_true, y_pred, average='micro')

Macro_f1_score = f1_score(y_true, y_pred, average='macro')
micro_f1_score = f1_score(y_true, y_pred, average='micro')

res = "%f\t%f\t%f\t%f\t%f\t%f\t%f"%(Accuracy,Macro_Average,micro_Average,Macro_recall,micro_recall,Macro_f1_score,micro_f1_score)

debug = 0







