# intention_detection
detect intention
## start
- download the [bert-chinese-model](https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip) and put it into /intention_detection/dataset/bert
---
-  download the [word2vec-chinese-model](https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.zh.zip) and put it into /intention_detection/dataset/
---
-  run the following files(self_train) to generate the model in \intention_detection\output

> main_bert_bilgru_softmax.py
    main_bert_bilstm_softmax.py
    main_bert_gru_softmax.py
    main_bert_lstm_softmax.py
    main_word2vec_bilgru_softmax.py
    main_word2vec_bilstm_softmax.py
    main_word2vec_gru_softmax.py
    main_word2vec_lstm_softmax.py

---
- run the above files(test) to test the model

- you can modify the dataset in the \intention_detection\dataset\test , but you must replace the space and so on(follow the format of the dataset files).

