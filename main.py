#!/usr/bin/python
# -*- coding: UTF-8 -*-

import yaml
import sys

# sys.setdefaultencoding("utf-8")
from sklearn.model_selection import train_test_split
import multiprocessing
import numpy as np
from keras.utils import np_utils
from gensim.models import KeyedVectors as Word2Vec
from gensim.corpora.dictionary import Dictionary

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.models import load_model
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Dropout, Activation
from keras.models import model_from_yaml
from sklearn.preprocessing import LabelEncoder

np.random.seed(1337)  # For Reproducibility
import jieba
import pandas as pd

# 设置最大递归层数 100w
sys.setrecursionlimit(1000000)
# 设置词向量维度
vocab_dim = 300
maxlen = 100
n_iterations = 1  # ideally more..
n_exposures = 10
window_size = 7
batch_size = 32
# 训练轮数
n_epoch = 15
input_length = 100
cpu_count = multiprocessing.cpu_count()


# 加载训练文件
def loadfile():
    fopen = open('D:/PyCharm 2019.2/projects/lstm_softmax/dataset/music', 'r', encoding='utf-8')
    music = []
    for line in fopen:
        music.append(line)

    fopen = open('D:/PyCharm 2019.2/projects/lstm_softmax/dataset/question', 'r',encoding='utf-8')
    question = []
    for line in fopen:
        question.append(line)

    fopen = open('D:/PyCharm 2019.2/projects/lstm_softmax/dataset/station', 'r',encoding='utf-8')
    station = []
    for line in fopen:
        station.append(line)

    combined = np.concatenate((music, question, station))
    # 定义三个一维数组 并进行初始化
    question_array = np.array([-1] * len(question), dtype=int)
    station_array = np.array([0] * len(station), dtype=int)
    music_array = np.array([1] * len(music), dtype=int)
    # y = np.concatenate((np.ones(len(station), dtype=int), np.zeros(len(music), dtype=int)),qabot_array[0])
    # 沿着水平方向将数组连接起来 返回一个数组
    y = np.hstack((music_array,question_array, station_array))
    print ("y is:")
    print (y.size)
    print ("combines is:")
    print (combined.size)
    return combined, y


# 对句子分词，并去掉换行符
def tokenizer(document):
    ''' Simple Parser converting each document to lower-case, then
        removing the breaks for new lines and finally splitting on the
        whitespace
    '''
    result_list = []
    for text in document:
        temp1 = ' '.join(list(jieba.cut(text.replace("\n","")))) # 正则去标点 [+\!\/_,$%^*(+\"\')]+|[:：+——()?【】“”！，。？、~@#￥%……&*（）／]+
        result_list.append(temp1.strip())
    return result_list


# 创建词语字典，并返回每个词语的索引，词向量，以及每个句子所对应的词语索引
def create_dictionaries(model=None,
                        combined=None):
    ''' Function does are number of Jobs:
        1- Creates a word to index mapping
        2- Creates a word to vector mapping
        3- Transforms the Training and Testing Dictionaries
        4- 返回所有词语的向量的拼接结果
    '''
    if (combined is not None) and (model is not None):
        gensim_dict = Dictionary()
        # keys
        gensim_dict.doc2bow(model.wv.vocab.keys(), allow_update=True)
        w2indx = {v: k + 1 for k, v in gensim_dict.items()}  # 所有频数超过10的词语的索引
        w2vec = {word: model[word] for word in w2indx.keys()}  # 所有频数超过10的词语的词向量

        def parse_dataset(combined):
            ''' Words become integers
            '''
            data = []
            for sentence in combined:
                new_txt = []
                sentences = sentence.split(' ')
                for word in sentences:
                    try:
                        #word = np.unicode(word, errors='ignore')
                        new_txt.append(w2indx[word])
                    except:
                        new_txt.append(0)
                data.append(new_txt)
            return data

        combined = parse_dataset(combined)
        # combined = sequence.pad_sequences(combined, maxlen=maxlen)  # 每个句子所含词语对应的索引，所以句子中含有频数小于10的词语，索引为0
        combined = sequence.pad_sequences(combined)  # 每个句子所含词语对应的索引，所以句子中含有频数小于10的词语，索引为0
        return w2indx, w2vec, combined
    else:
        print ('No data provided...')


# 创建词语字典，并返回每个词语的索引，词向量，以及每个句子所对应的词语索引
def word2vec_train(combined):
    # 加载word2vec 模型
    model = Word2Vec.load_word2vec_format('D:\\PyCharm 2019.2\\projects\\lstm_softmax\\dataset\\wiki.zh.vec',binary=False)
    index_dict, word_vectors, combined = create_dictionaries(model=model, combined=combined)
    return index_dict, word_vectors, combined


def get_data(index_dict, word_vectors, combined, y):
    # 获取句子的向量
    n_symbols = len(index_dict) + 1  # 所有单词的索引数，频数小于10的词语索引为0，所以加1
    # vocab_dim 词向量的维度 建立一个 单词数量 和 向量维度 矩阵 纵坐标可代替向量索引
    embedding_weights = np.zeros((n_symbols, vocab_dim))  # 索引为0的词语，词向量全为0
    for word, index in index_dict.items():  # 从索引为1的词语开始，对每个词语对应其词向量
        embedding_weights[index] = word_vectors[word]
    # 获取训练结合 和 验证集合
    x_train, x_test, y_train, y_test = train_test_split(combined, y, test_size=0.2)
    # encode class values as integers
    # LabelEncoder LabelEncoder可以将标签分配一个0—n_classes-1之间的编码 将各种标签分配一个可数的连续编号： 例如：
    # out = ['我', '是', '中', '国', '人', '我', '是', '是', '好', '人']
    # outy = LabelEncoder().fit_transform(out)
    # [4 5 0 2 1 4 5 5 3 1]

    # train_test_split之后是乱序
    encoder = LabelEncoder()
    encoded_y_train = encoder.fit_transform(y_train)
    encoded_y_test = encoder.fit_transform(y_test)
    # convert integers to dummy variables (one hot encoding)
    # 进行多分类的矩阵的创建 y:0 1 2 (三分类) x:0 1 2 3 句子id 对应的(x,y)=1属于x属于当前这个类y oneHot编码
    y_train = np_utils.to_categorical(encoded_y_train)
    y_test = np_utils.to_categorical(encoded_y_test)
    print (x_train.shape, y_train.shape)
    return n_symbols, embedding_weights, x_train, y_train, x_test, y_test


##定义网络结构
def train_lstm(n_symbols, embedding_weights, x_train, y_train, x_test, y_test):
    nb_classes = 3
    print ('Defining a Simple Keras Model...')
    ## 定义基本的网络结构
    model = Sequential()  # or Graph or whatever
    ## 对于LSTM 变长的文本使用Embedding 将其变成指定长度的向量
    # 查看 : https://keras.io/zh/layers/embeddings/
    # 嵌入层
    # 将正整数（索引值）转换为固定尺寸的稠密向量。 例如： [[4], [20]] -> [[0.25, 0.1], [0.6, -0.2]]
    # 该层只能用作模型中的第一层。
    model.add(Embedding(output_dim=vocab_dim, # 词向量的维度
                        input_dim=n_symbols, # 词汇表的大小
                        mask_zero=True, # 是否把 0 看作为一个应该被遮蔽的特殊的 "padding" 值
                        weights=[embedding_weights], #
                        input_length=len(embedding_weights[0])))  # 输入序列的长度
                        # input_length=input_length))  # 输入序列的长度
    ## 使用单层LSTM 输出的向量维度是50，输入的向量维度是vocab_dim,激活函数relu
    # 循环层
    model.add(LSTM(output_dim=50, activation='relu', inner_activation='hard_sigmoid'))
    # 放弃层
    model.add(Dropout(0.5))
    # 在这里外接softmax，进行最后的3分类 输出层
    model.add(Dense(output_dim=nb_classes, input_dim=50, activation='softmax'))
    print ('Compiling the Model...')
    # loss: 字符串（目标函数名）或目标函数  激活函数使用的是adam    metrics 在训练和测试期间的模型评估标准
    model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])

    print ("Train...")
    print (y_train)
    # epochs: 整数。训练模型迭代轮次 validation_data:  用来评估损失，  batch_size: 整数或 None。每次梯度更新的样本数
    # verbose: 0, 1 或 2。日志显示模式。 0 = 安静模式, 1 = 进度条, 2 = 每轮一行。
    model.fit(x_train, y_train, batch_size=batch_size, epochs=n_epoch, verbose=1, validation_data=(x_test, y_test))
    print ("Evaluate...")
    score = model.evaluate(x_test, y_test,batch_size=batch_size)
    #保存模型
    yaml_string = model.to_yaml()
    with open('output/lstm_koubei.yml', 'w') as outfile:
        outfile.write(yaml.dump(yaml_string, default_flow_style=True))
    model.save('output/lstm_koubei.h5')
    print ('Test score:', score)


# 训练模型，并保存
def train():
    print ('Loading Data...')
    combined, y = loadfile()
    print (len(combined), len(y))
    print ('Tokenising...')
    combined = tokenizer(combined)
    print ('Training a Word2vec model...')
    index_dict, word_vectors, combined = word2vec_train(combined)
    print ('Setting up Arrays for Keras Embedding Layer...')
    n_symbols, embedding_weights, x_train, y_train, x_test, y_test = get_data(index_dict, word_vectors, combined, y)
    print (x_train.shape, y_train.shape)
    train_lstm(n_symbols, embedding_weights, x_train, y_train, x_test, y_test)


# 训练模型，并保存
def self_train():
    print ('Loading Data...')
    combined, y = loadfile()
    print (len(combined), len(y))
    print ('Tokenising...')
    combined = tokenizer(combined)
    print ('Training a Word2vec model...')
    index_dict, word_vectors, combined = word2vec_train(combined)
    print ('Setting up Arrays for Keras Embedding Layer...')
    n_symbols, embedding_weights, x_train, y_train, x_test, y_test = get_data(index_dict, word_vectors, combined, y)
    print (x_train.shape, y_train.shape)
    train_lstm(n_symbols, embedding_weights, x_train, y_train, x_test, y_test)


def input_transform(string):
    temp1 = ' '.join(list(jieba.cut(string.replace("\n", ""))))  # 正则去标点 [+\!\/_,$%^*(+\"\')]+|[:：+——()?【】“”！，。？、~@#￥%……&*（）／]+
    words = temp1.strip()
    tmp_list = []
    tmp_list.append(words)
    # words=np.array(tmp_list).reshape(1,-1)
    model = Word2Vec.load_word2vec_format('D:\\PyCharm 2019.2\\projects\\lstm_softmax\\dataset\\wiki.zh.vec',binary=False)
    index_dict, word_vectors, combined = create_dictionaries(model, tmp_list)
    return combined


if __name__ == '__main__':
    self_train()
    # inp = input_transform("我喜欢听一些电台广播") #1
    #
    # model_save_path = 'D:\\PyCharm 2019.2\\projects\\lstm_softmax\\output\\lstm_koubei.h5'
    # lstm_model = load_model(model_save_path)
    #
    # res = lstm_model.predict(inp)
    #
    # debuf= 0

    # y_predict = lstm_model.predict(x)
    # label_dict = {v: k for k, v in output_dictionary.items()}
    # print('输入语句: %s' % sent)
    # print('情感预测结果: %s' % label_dict[np.argmax(y_predict)])