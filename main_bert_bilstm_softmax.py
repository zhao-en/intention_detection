#!/usr/bin/python
# -*- coding: UTF-8 -*-

import yaml
import sys

from keras.engine.saving import load_model
from keras.layers import Bidirectional
from sklearn.model_selection import train_test_split
import numpy as np
from keras.utils import np_utils
from gensim.models import KeyedVectors as Word2Vec
from keras_bert import get_base_dict, get_model, gen_batch_inputs, load_trained_model_from_checkpoint
from gensim.corpora.dictionary import Dictionary
from bert_serving.client import BertClient
import matplotlib.pyplot as plt
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Dropout
from sklearn.preprocessing import LabelEncoder

np.random.seed(1337)
import jieba

# 设置最大递归层数 100w
sys.setrecursionlimit(1000000)
# 设置词向量维度
vocab_dim = 768
# 设置 每次梯度更新的样本数
batch_size = 32
# 训练总轮数
n_epoch = 8
# 词嵌如pad之后的长度
input_length = 100
# 分类的数量
nb_classes = 5

# 设置模型参数 crf / softmax
set_activation='softmax'

vocab = []

# 1170/1170 [==============================] - 2s 2ms/step - loss: 0.0461 - acc: 0.9829 - val_loss: 1.5322 - val_acc: 0.7611

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
    y = np.hstack(
        (search_restaurant_array, support_info_array, ask_question_array, change_result_array, thanks_bye_array))
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
    bc = BertClient()
    if (combined is not None) and (model is not None):
        gensim_dict = Dictionary()
        # 获取keys集合,字典的单词集合
        words = []
        for sentence in combined:
            sentences = sentence.split(' ')
            for word in sentences:
                if word != '':
                    words.append(word)
        gensim_dict.doc2bow(words, allow_update=True)
        # 获取word_index=>index集合
        w2indx = {v: k + 1 for k, v in gensim_dict.items()}

# debug
        # for word in w2indx.keys():
        #     vc = [word]
        #     v = bc.encode([word])
        #     t = bc.encode([word])[0]
        #     c = 0

        # 获取word=>词向量集合
        w2vec = {word: bc.encode([word])[0] for word in w2indx.keys()}

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
    # model = Word2Vec.load_word2vec_format('D:\\PyCharm 2019.2\\projects\\lstm_softmax\\dataset\\wiki.zh.vec',binary=False)
    # model = load_trained_model_from_checkpoint(
    #     'D:\\PyCharm 2019.2\\projects\\lstm_softmax\\dataset\\bert\\bert_config.json',
    #     'D:\\PyCharm 2019.2\\projects\\lstm_softmax\\dataset\\bert\\bert_model.ckpt',
    #     training=False,
    #     seq_len=None)
    model = ""
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
    model.add(Bidirectional(LSTM(output_dim=50, activation='relu', inner_activation='hard_sigmoid')))

    # 放弃层
    model.add(Dropout(0.2))

    # 在这里外接softmax，进行最后的3分类 输出层
    model.add(Dense(units=nb_classes, input_dim=50, activation=set_activation))

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

    f = open('D:/PyCharm 2019.2/projects/lstm_softmax/output/bert_bilstm_softmax_history', 'w')
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
    with open('output/bert_bilstm_softmax.yml', 'w') as outfile:
        outfile.write(yaml.dump(yaml_string, default_flow_style=True))
    model.save('output/bert_bilstm_softmax.h5')


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
    # index_dict, word_vectors, combined = create_dictionaries(model="", combined=testdata2)

    # fopen = open('D:/PyCharm 2019.2/projects/lstm_softmax/output/bert_lstm_softmax_vocab', 'r', encoding='utf-8')
    # dict = fopen.read()
    # fopen.close()

    dict = {'100': 1, '25': 2, 'COTE': 3, 'Canapes': 4, 'CherryHinton': 5, 'F': 6, 'Prezzo': 7, 'TanhBinh': 8, 'cocum': 9, 'nandos': 10, '一下': 11, '一个': 12, '一些': 13, '一切': 14, '一天': 15, '一家': 16, '一点': 17, '一种': 18, '一般': 19, '一起': 20, '一部分': 21, '万分': 22, '三家': 23, '三明治': 24, '上海': 25, '下': 26, '下次': 27, '不': 28, '不了': 29, '不介意': 30, '不会': 31, '不到': 32, '不同': 33, '不在乎': 34, '不好': 35, '不想': 36, '不感兴趣': 37, '不换个': 38, '不是': 39, '不用': 40, '不管': 41, '不能': 42, '不花': 43, '不行': 44, '不要': 45, '不过': 46, '不远': 47, '不错': 48, '专门': 49, '世界': 50, '东': 51, '东侧': 52, '东区': 53, '东方': 54, '东西': 55, '东边': 56, '东部': 57, '丢失': 58, '两个': 59, '两家': 60, '两种': 61, '中': 62, '中国': 63, '中心': 64, '中文': 65, '中等': 66, '中间': 67, '中餐': 68, '中餐馆': 69, '串串': 70, '丹麦': 71, '为': 72, '为了': 73, '主菜': 74, '举办活动': 75, '么': 76, '也': 77, '乡村': 78, '买': 79, '买点': 80, '了': 81, '了解': 82, '事实上': 83, '二二': 84, '亚洲': 85, '产品': 86, '亲爱': 87, '人': 88, '什么': 89, '什么样': 90, '仅仅': 91, '仅此而已': 92, '今天': 93, '他': 94, '他们': 95, '以下': 96, '以及': 97, '价位': 98, '价格': 99, '价格低': 100, '价格便宜': 101, '价格合理': 102, '价格昂贵': 103, '价格比': 104, '任何': 105, '任何人': 106, '任意': 107, '优惠': 108, '会': 109, '会场': 110, '会见': 111, '传统': 112, '伦敦': 113, '伯里': 114, '但': 115, '位于': 116, '位置': 117, '你': 118, '你们': 119, '你好': 120, '使用': 121, '供应': 122, '便宜': 123, '俄罗斯': 124, '信息': 125, '偏好': 126, '元': 127, '先': 128, '先生': 129, '全部': 130, '公司': 131, '兰州': 132, '关系': 133, '其中': 134, '其他': 135, '内': 136, '再': 137, '再会': 138, '再来一个': 139, '再次': 140, '再聊': 141, '再见': 142, '冒': 143, '写': 144, '写明': 145, '农家': 146, '决定': 147, '几个': 148, '几家': 149, '出名': 150, '创意': 151, '利亚': 152, '别的': 153, '到': 154, '到处': 155, '剑桥': 156, '加勒比': 157, '加勒比海': 158, '加拿大': 159, '加泰罗尼亚': 160, '匈牙利': 161, '北方': 162, '北美': 163, '北边': 164, '北部': 165, '区': 166, '区域': 167, '区间': 168, '卖': 169, '南': 170, '南侧': 171, '南区': 172, '南城': 173, '南多': 174, '南方': 175, '南边': 176, '南部': 177, '印尼': 178, '印度': 179, '印度人': 180, '印第安': 181, '印第安人': 182, '厄立特里亚': 183, '去': 184, '又': 185, '发': 186, '发现': 187, '口味': 188, '古巴': 189, '另一家': 190, '只': 191, '只想': 192, '只有': 193, '只要': 194, '叫': 195, '可以': 196, '可爱': 197, '可能': 198, '号': 199, '号码': 200, '吃': 201, '吃火锅': 202, '吃点': 203, '吃饭': 204, '同一': 205, '名字': 206, '名称': 207, '后': 208, '后会有期': 209, '向': 210, '吗': 211, '吧': 212, '听': 213, '呆': 214, '告诉': 215, '告辞': 216, '周围': 217, '味道': 218, '和': 219, '咖喱': 220, '哈': 221, '哦': 222, '哪': 223, '哪个': 224, '哪些': 225, '哪些地方': 226, '哪家': 227, '哪有': 228, '哪里': 229, '唐平': 230, '啊': 231, '喜欢': 232, '喝咖啡': 233, '喽': 234, '嗨': 235, '嗯': 236, '嘿': 237, '四川': 238, '四川菜': 239, '回锅肉': 240, '因': 241, '国际': 242, '土耳其': 243, '圣约翰': 244, '圣诞': 245, '在': 246, '在城镇': 247, '地中海': 248, '地区': 249, '地址': 250, '地方': 251, '地点': 252, '场地': 253, '址': 254, '城东': 255, '城中心': 256, '城北': 257, '城南': 258, '城市': 259, '城西': 260, '城里': 261, '城镇': 262, '基的': 263, '塔斯卡': 264, '填上': 265, '填写': 266, '墨西哥': 267, '墨西哥人': 268, '多么': 269, '多少': 270, '多斯': 271, '大': 272, '大厦': 273, '太': 274, '太多': 275, '太好了': 276, '太有钱': 277, '太棒了': 278, '太贵': 279, '太远': 280, '奥地利': 281, '奶茶': 282, '好': 283, '好吃': 284, '好幸': 285, '如何': 286, '如果': 287, '威尔士': 288, '威尼斯': 289, '学校': 290, '它': 291, '它们': 292, '安纳托': 293, '安静': 294, '完美': 295, '定': 296, '实惠': 297, '实际上': 298, '家': 299, '家常菜': 300, '对': 301, '对不起': 302, '寻常': 303, '寻找': 304, '将': 305, '小': 306, '小吃': 307, '小吃店': 308, '小孩子': 309, '小酒馆': 310, '尝尝': 311, '就': 312, '就是': 313, '就行了': 314, '尼': 315, '川菜': 316, '川菜馆': 317, '已经': 318, '巴拿马': 319, '巴斯克': 320, '巴西': 321, '巷': 322, '市': 323, '市中心': 324, '市区': 325, '布卢姆': 326, '希望': 327, '希腊': 328, '帮': 329, '帮助': 330, '并': 331, '幸运星': 332, '广东': 333, '应该': 334, '店': 335, '建议': 336, '建设': 337, '开': 338, '开元': 339, '式': 340, '弗兰基': 341, '当然': 342, '往东': 343, '很': 344, '很多': 345, '很感兴趣': 346, '很穷': 347, '很贵': 348, '很酷': 349, '很长': 350, '得到': 351, '德克士': 352, '德国': 353, '心有': 354, '必胜客': 355, '必须': 356, '怎么': 357, '怎么样': 358, '急需': 359, '您': 360, '您好': 361, '情侣': 362, '情况': 363, '想': 364, '想换': 365, '想要': 366, '愉快': 367, '意大利': 368, '意大利语': 369, '感兴趣': 370, '感激不尽': 371, '感谢': 372, '感谢您': 373, '愿意': 374, '成华区': 375, '成都': 376, '我': 377, '我们': 378, '我会': 379, '我尼': 380, '我能': 381, '我要': 382, '我远': 383, '我金锅': 384, '或者': 385, '房源': 386, '房间': 387, '所': 388, '所在': 389, '所有': 390, '打包': 391, '打听一下': 392, '打电话': 393, '托斯卡纳': 394, '找': 395, '找个': 396, '找些': 397, '找到': 398, '找家': 399, '找点': 400, '找西': 401, '把': 402, '披萨': 403, '拉': 404, '拉扎': 405, '拉拉': 406, '拉米': 407, '拉西亚': 408, '拉面': 409, '拿到': 410, '指': 411, '换': 412, '换成': 413, '排队': 414, '接受': 415, '推荐': 416, '提个': 417, '提供': 418, '提出': 419, '搜索': 420, '搞定': 421, '摩洛哥': 422, '擅长': 423, '斯': 424, '斯堪的纳维亚': 425, '新加坡': 426, '新疆': 427, '旁边': 428, '无': 429, '无关紧要': 430, '无所谓': 431, '日常': 432, '日本': 433, '日本料理': 434, '昂贵': 435, '明天': 436, '春熙路': 437, '是': 438, '是否': 439, '晚上': 440, '晚安': 441, '晚餐': 442, '更': 443, '曼谷': 444, '最': 445, '最好': 446, '最贵': 447, '最近': 448, '有': 449, '有个': 450, '有机': 451, '有没有': 452, '有用': 453, '有缘': 454, '朋友': 455, '本尼': 456, '本市': 457, '机会': 458, '条件': 459, '来说': 460, '查理': 461, '栀子': 462, '格拉夫': 463, '欢迎': 464, '欧式': 465, '欧洲': 466, '正在': 467, '正宗': 468, '每': 469, '每个': 470, '比利时': 471, '比萨': 472, '比较': 473, '沙县': 474, '没': 475, '没事': 476, '没关系': 477, '没有': 478, '河南': 479, '法国': 480, '法国菜': 481, '法式': 482, '泛亚': 483, '波兰': 484, '波斯': 485, '泰': 486, '泰国': 487, '泰国菜': 488, '泰姬陵': 489, '洛奇': 490, '海': 491, '海鲜': 492, '涂鸦': 493, '消息': 494, '混合': 495, '清真': 496, '澄清': 497, '澳大': 498, '澳大利亚': 499, '火锅': 500, '火锅店': 501, '炸酱面': 502, '炸鱼': 503, '点': 504, '烧': 505, '烧烤': 506, '烧烤店': 507, '烩面': 508, '热烈欢迎': 509, '煲': 510, '爱尔兰': 511, '牙买加': 512, '牛排': 513, '牛排馆': 514, '物品': 515, '特别': 516, '特定': 517, '特殊': 518, '犹太': 519, '王子': 520, '环境': 521, '现代': 522, '现在': 523, '瑞典': 524, '瑞士': 525, '生活': 526, '用': 527, '电话': 528, '电话号码': 529, '的': 530, '的话': 531, '皇家': 532, '皮帕沙': 533, '相信': 534, '相同': 535, '相当': 536, '看': 537, '看看': 538, '真正': 539, '真的': 540, '知道': 541, '确定': 542, '祝': 543, '祝你好运': 544, '祝你快乐': 545, '祝您': 546, '离': 547, '种': 548, '种类': 549, '科西嘉': 550, '符合': 551, '第三家': 552, '米粉': 553, '米线': 554, '米饭': 555, '类别': 556, '类型': 557, '粤菜': 558, '粤菜馆': 559, '系统': 560, '素食': 561, '经历': 562, '经营': 563, '给': 564, '绵阳': 565, '缩小': 566, '罗尼亚': 567, '罗马尼亚': 568, '罚款': 569, '羊肉汤': 570, '美好': 571, '美食': 572, '美食家': 573, '者': 574, '而': 575, '而且': 576, '聊天': 577, '联系方式': 578, '联系电话': 579, '肖普': 580, '肖普之家': 581, '胡辣汤': 582, '能': 583, '能换': 584, '芬迪顿': 585, '花': 586, '苏格兰': 587, '英国': 588, '英国人': 589, '英式': 590, '英语': 591, '范围': 592, '莫萨': 593, '菜': 594, '菜式': 595, '菜肴': 596, '菜馆': 597, '萨拉': 598, '葡萄牙': 599, '葡萄牙语': 600, '薯条': 601, '融合': 602, '街': 603, '街道': 604, '表示': 605, '西': 606, '西侧': 607, '西区': 608, '西方': 609, '西班牙': 610, '西端': 611, '西贡': 612, '西边': 613, '西部': 614, '西餐': 615, '西餐厅': 616, '要': 617, '要求': 618, '要点': 619, '见': 620, '见到': 621, '觉得': 622, '记下': 623, '讲': 624, '设拉子': 625, '评价': 626, '评分': 627, '试试': 628, '语': 629, '说': 630, '请': 631, '请换': 632, '请问': 633, '谢平': 634, '谢谢': 635, '谢谢您': 636, '豫园': 637, '贝都': 638, '贵': 639, '贵不贵': 640, '贵点': 641, '起来': 642, '越南': 643, '越南语': 644, '距离': 645, '较贵': 646, '输入': 647, '辣椒': 648, '达芬奇': 649, '过': 650, '过去': 651, '过得': 652, '近': 653, '还': 654, '还会': 655, '还好': 656, '还有': 657, '还要': 658, '这': 659, '这个': 660, '这些': 661, '这家': 662, '这是': 663, '这样': 664, '这样的话': 665, '这里': 666, '进': 667, '进一步': 668, '适中': 669, '适合': 670, '选': 671, '选择': 672, '选项': 673, '通': 674, '那': 675, '那个': 676, '那么': 677, '那家': 678, '那就好': 679, '那里': 680, '邮政编码': 681, '邮编': 682, '都': 683, '酒吧': 684, '酒店': 685, '里': 686, '里面': 687, '重新': 688, '重要': 689, '金锅': 690, '钱': 691, '镇': 692, '镇上': 693, '镇北有': 694, '镇子': 695, '问': 696, '问题': 697, '阳光': 698, '阿富汗': 699, '阿里巴巴': 700, '附上': 701, '附近': 702, '陈': 703, '限': 704, '随便': 705, '难': 706, '需不需要': 707, '需要': 708, '非常': 709, '非常感激': 710, '非常感谢': 711, '非洲': 712, '面': 713, '面积': 714, '面食': 715, '面馆': 716, '韦斯特': 717, '韩国': 718, '韩国泡菜': 719, '韩国菜': 720, '顿': 721, '风格': 722, '食品': 723, '食品店': 724, '食物': 725, '餐': 726, '餐厅': 727, '餐馆': 728, '饭店': 729, '饭菜': 730, '饺子': 731, '饿': 732, '馒头': 733, '首尔': 734, '首选项': 735, '香料': 736, '香锅': 737, '马来西亚': 738, '高': 739, '高档': 740, '鸡': 741, '鸡公': 742, '麦当劳': 743, '麻辣': 744, '黄焖': 745, '黎巴嫩': 746, '黑': 747, '，': 748}
    combined2 = [[]]
    testdata2 = testdata2[0].split(" ")
    for w in testdata2:
        if w in dict:
            combined2[0].append(dict[w])
        else:
            combined2[0].append(0)
    combined2 = sequence.pad_sequences(list(combined2),maxlen=23)
    return combined2

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
    lstm_model = load_model('D:\\PyCharm 2019.2\\projects\\lstm_softmax\\output\\bert_bilstm_softmax.h5')
    for path in paths:
        fopen = open(path, 'r', encoding='utf-8')
        count = [0, 0, 0, 0, 0]
        for line in fopen:
            temp = input_transform([line])
            res = lstm_model.predict(temp)
            max = 0
            max_index = -1
            for index, val in enumerate(res[0]):
                if val > max:
                    max = val
                    max_index = index
            count[max_index] = count[max_index] + 1
        result.append(count)
        fopen.close()

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
