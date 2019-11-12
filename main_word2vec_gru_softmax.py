#!/usr/bin/python
# -*- coding: UTF-8 -*-

import yaml
import sys

from keras.engine.saving import load_model
from sklearn.model_selection import train_test_split
import numpy as np
from keras.utils import np_utils
from gensim.models import KeyedVectors as Word2Vec
from gensim.corpora.dictionary import Dictionary
from keras.utils import np_utils, plot_model
import matplotlib.pyplot as plt

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from keras.layers.core import Dense, Dropout
from sklearn.preprocessing import LabelEncoder

np.random.seed(1337)
import jieba

# 设置最大递归层数 100w
sys.setrecursionlimit(1000000)
# 设置词向量维度
vocab_dim = 300
# 设置 每次梯度更新的样本数
batch_size = 32
# 训练总轮数
n_epoch = 21
# 词嵌如pad之后的长度
input_length = 100
# 分类的数量
nb_classes = 5

# 设置模型参数 crf / softmax
set_activation='softmax'

# 1170/1170 [==============================] - 5s 4ms/step - loss: 0.1090 - acc: 0.9632 - val_loss: 1.3972 - val_acc: 0.7167

# 加载训练文件
def loadfile():
    fopen = open('D:/PyCharm 2019.2/projects/lstm_softmax/dataset/1查询餐厅_2', 'r', encoding='utf-8')
    search_restaurant = []
    for line in fopen:
        search_restaurant.append(line)

    fopen = open('D:/PyCharm 2019.2/projects/lstm_softmax/dataset/2提供信息_2', 'r', encoding='utf-8')
    support_info = []
    for line in fopen:
        support_info.append(line)

    fopen = open('D:/PyCharm 2019.2/projects/lstm_softmax/dataset/3询问问题_2', 'r', encoding='utf-8')
    ask_question = []
    for line in fopen:
        ask_question.append(line)

    fopen = open('D:/PyCharm 2019.2/projects/lstm_softmax/dataset/4换一个结果_2', 'r', encoding='utf-8')
    change_result = []
    for line in fopen:
        change_result.append(line)

    fopen = open('D:/PyCharm 2019.2/projects/lstm_softmax/dataset/910欢迎感谢再见_2', 'r', encoding='utf-8')
    thanks_bye = []
    for line in fopen:
        thanks_bye.append(line)

    combined = np.concatenate((search_restaurant, support_info, ask_question, change_result, thanks_bye))

    # 定义三个一维数组 并进行初始化
    search_restaurant_array = np.array([-2] * len(search_restaurant), dtype=int)
    support_info_array = np.array([-1] * len(support_info), dtype=int)
    ask_question_array = np.array([0] * len(ask_question), dtype=int)
    change_result_array = np.array([1] * len(change_result), dtype=int)
    thanks_bye_array = np.array([2] * len(thanks_bye), dtype=int)

    # 沿着水平方向将数组连接起来 返回一个数组
    y = np.hstack((search_restaurant_array, support_info_array, ask_question_array, change_result_array, thanks_bye_array))
    return combined, y


# 对句子分词，并去掉换行符
# 预处理 正则去标点 [+\!\/_,$%^*(+\"\')]+|[:：+——()?【】“”！，。？、~@#￥%……&*（）／]+
def tokenizer(document):
    result_list = []
    for text in document:
        temp1 = ' '.join(
            list(jieba.cut(text.replace("\n", ""))))
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
        # 获取keys集合,字典的单词集合
        gensim_dict.doc2bow(model.wv.vocab.keys(), allow_update=True)
        # 获取word_index=>index集合
        w2indx = {v: k + 1 for k, v in gensim_dict.items()}
        # 获取word=>词向量集合
        w2vec = {word: model[word] for word in w2indx.keys()}

        def parse_dataset(combined):
            ''' Words become integers
            '''
            data = []
            for sentence in combined:
                new_txt = []
                sentences = sentence.split(' ')
                for word in sentences:
                    try:
                        new_txt.append(w2indx[word])
                    except:
                        new_txt.append(0)
                data.append(new_txt)
            return data

        combined = parse_dataset(combined)
        # pad 补上0
        combined = sequence.pad_sequences(combined)
        global input_length
        input_length = len(combined[0])
        return w2indx, w2vec, combined
    else:
        print('error: 模型或者和并集合combined 为空')


# 创建词语字典，并返回每个词语的索引，词向量，以及每个句子所对应的词语索引
def word2vec_load(combined):
    # 加载word2vec 模型
    model = Word2Vec.load_word2vec_format('D:\\PyCharm 2019.2\\projects\\lstm_softmax\\dataset\\wiki.zh.vec',binary=False)
    index_dict, word_vectors, combined = create_dictionaries(model=model, combined=combined)
    return index_dict, word_vectors, combined


# 处理数据 获得嵌入层输入
def get_data(index_dict, word_vectors, combined, y):
    # 获取字典索引数量 + 1
    n_symbols = len(index_dict) + 1

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

    # 进行多分类的矩阵的创建 y:0 1 2 (三分类) x:0 1 2 3 句子id 对应的(x,y)=1属于x属于当前这个类y oneHot编码
    y_train = np_utils.to_categorical(encoded_y_train)
    y_test = np_utils.to_categorical(encoded_y_test)

    return n_symbols, embedding_weights, x_train, y_train, x_test, y_test

def build_model(n_symbols, embedding_weights):
    model = Sequential()
    # 嵌入层
    # 将正整数（索引值）转换为固定尺寸的稠密向量。 例如： [[4], [20]] -> [[0.25, 0.1], [0.6, -0.2]]
    # 该层只能用作模型中的第一层。
    model.add(Embedding(output_dim=vocab_dim,  # 词向量的维度
                        input_dim=n_symbols,  # 词汇表的大小
                        mask_zero=True,  # 是否把 0 看作为一个应该被遮蔽的特殊的 "padding" 值
                        weights=[embedding_weights],  #
                        input_length=input_length))  # 输入序列的长度

    # 循环层
    # 使用单层LSTM 输出的向量维度是50，输入的向量维度是vocab_dim,激活函数relu
    model.add(GRU(output_dim=50, activation='relu', inner_activation='hard_sigmoid'))

    # 放弃层
    model.add(Dropout(0.2))

    # 在这里外接softmax，进行最后的3分类 输出层
    model.add(Dense(output_dim=nb_classes, input_dim=50, activation=set_activation))

    return model

# 定义模型结构 并 进行训练
def train_lstm(model, x_train, y_train, x_test, y_test):


    # 编译模型
    print('info:正在编译模型...')
    # loss: 字符串（目标函数名）或目标函数  激活函数使用的是adam    metrics 在训练和测试期间的模型评估标准
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    print("info:开始训练...")
    # epochs: 整数。训练模型迭代轮次 validation_data:  用来评估损失，  batch_size: 整数或 None。每次梯度更新的样本数
    # verbose: 0, 1 或 2。日志显示模式。 0 = 安静模式, 1 = 进度条, 2 = 每轮一行。
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=n_epoch, verbose=1, validation_data=(x_test, y_test))

    f = open('D:/PyCharm 2019.2/projects/lstm_softmax/output/word2vec_gru_softmax_history', 'w')
    f.write(str(history.history))
    f.close()

    # -------------------------------------------------------
    # 绘制训练 & 验证的准确率值
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    # 绘制训练 & 验证的损失值
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()


    print("info:验证集验证模型...")
    score = model.evaluate(x_test, y_test, batch_size=batch_size)
    print('info:验证得分(loss ,  acc)：======= ', score)

    # 保存模型
    print("info:保存模型(.h5/yml)...")
    yaml_string = model.to_yaml()
    with open('output/word2vec_gru_softmax.yml', 'w') as outfile:
        outfile.write(yaml.dump(yaml_string, default_flow_style=True))
    model.save('output/word2vec_gru_softmax.h5')


# 训练模型，并保存
def self_train():
    print("step1:加载文件数据...")
    combined, y = loadfile()
    print("校对: 句子向量长度" , len(combined), "类别向量长度=" , len(y))

    print("step2:标注数据处理...")
    combined = tokenizer(combined)

    print('step3:加载Word2Vec并生成索引文件...')
    index_dict, word_vectors, combined = word2vec_load(combined)

    print('step4:嵌入层数据处理...')
    n_symbols, embedding_weights, x_train, y_train, x_test, y_test = get_data(index_dict, word_vectors, combined, y)
    print("校对,训练数据维度=" , x_train.shape, "目标数据维度=" , y_train.shape)

    print("step5:构建模型...")
    model = build_model( n_symbols, embedding_weights)

    print('step6:开始训练模型...')
    train_lstm(model, x_train, y_train, x_test, y_test)


# 模型测试
def input_transform(testdata):
    testdata2 = []
    for str in testdata:
        testdata2.append(' '.join(list(jieba.cut(str.replace("\n", "")))).strip())
    model = Word2Vec.load_word2vec_format('D:\\PyCharm 2019.2\\projects\\lstm_softmax\\dataset\\wiki.zh.vec',binary=False)
    index_dict, word_vectors, combined = create_dictionaries(model, testdata2)
    combined = sequence.pad_sequences(combined,maxlen=23)
    return combined

# 测试集合
def test():
    paths = [
        'D:/PyCharm 2019.2/projects/lstm_softmax/dataset/test/1查询餐厅test',
        'D:/PyCharm 2019.2/projects/lstm_softmax/dataset/test/2提供信息test',
        'D:/PyCharm 2019.2/projects/lstm_softmax/dataset/test/3询问问题test',
        'D:/PyCharm 2019.2/projects/lstm_softmax/dataset/test/4换一个结果test',
        'D:/PyCharm 2019.2/projects/lstm_softmax/dataset/test/910欢迎感谢再见test'
    ]
    # 五个意图统计
    result = []
    lstm_model = load_model('D:\\PyCharm 2019.2\\projects\\lstm_softmax\\output\\word2vec_gru_softmax.h5')
    for path in paths:
        fopen = open(path, 'r', encoding='utf-8')
        testdata = []
        for line in fopen:
            testdata.append(line)
        fopen.close()
        res = lstm_model.predict(input_transform(testdata))
        # 计算准确率
        count = [0,0,0,0,0]
        for list in res:
            max = 0
            max_index = 0
            for index , val in enumerate(list):
                if val > max:
                    max = val
                    max_index = index
            count[max_index] = count[max_index] + 1
        result.append(count)

    print(result)

    debuf= 0

if __name__ == '__main__':
    #self_train()
    test()
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
