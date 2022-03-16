#! -*- coding:utf-8 -*-
# 情感分析例子，利用MLM做 Zero-Shot/Few-Shot/Semi-Supervised Learning

import numpy as np
from bert4keras.backend import keras, K
from bert4keras.layers import Loss
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open
import os
# 选择使用第几张GPU卡，'0'为第一张
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

num_classes = 2
maxlen = 128
batch_size = 16
epochs = 5

# config_path = './models/uncased_L-12_H-768_A-12/bert_config.json'
# checkpoint_path = './models/uncased_L-12_H-768_A-12/bert_model.ckpt'
# dict_path = './models/uncased_L-12_H-768_A-12/vocab.txt'

config_path = './models/cased_L-24_H-1024_A-16/bert_config.json'
checkpoint_path = './models/cased_L-24_H-1024_A-16/bert_model.ckpt'
dict_path = './models/cased_L-24_H-1024_A-16/vocab.txt'


def load_data(filename):
    """加载数据
    单条格式：(文本, 标签id)
    """
    D = []
    i = 1
    with open(filename, encoding='utf-8') as f:
        for l in f:
            if i == 1: # 跳过数据第一行
                i = 2
            else:
                text,label = l.strip().split('\t')
                D.append((text,int(label)))
    return D

# 加载数据集
train_data = load_data('./datasets/SST-2/train.tsv')
valid_data = load_data('./datasets/SST-2/dev.tsv')
# test_data = load_data('./datasets/SST-2/test.tsv')

# 模拟标注和非标注数据
num_labeled = 32 # 标注数据的个数
train_data = train_data[:num_labeled]   # few-shot 带标签的数据
# unlabeled_data = [(t, 2) for t, l in train_data[num_labeled:]]  # 把部分数据的标签去掉（改为2）
# train_data = train_data + unlabeled_data  # 加上无标签的数据

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)

# 对应的任务描述
prompt = u'It was - .'
is_pre = 0
pos_id = tokenizer.token_to_id(u'great')
neg_id = tokenizer.token_to_id(u'terrible')

class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random= False):
        batch_token_ids, batch_segment_ids, batch_output_ids = [], [], []
        for is_end, (text, label) in self.sample(random):
            if label != 2:
                if is_pre:
                    token_ids, segment_ids = tokenizer.encode(prompt, text, maxlen=maxlen)
                else:
                    token_ids, segment_ids = tokenizer.encode(text, prompt, maxlen=maxlen)
            source_ids, target_ids = token_ids[:], token_ids[:]
            mask_idx = source_ids.index(102)+3  # ⭐️ 定位[mask]的位置 [CLS]: 101 [SEP]: 102, 得基于prompt来修改
            if label == 0:
                source_ids[mask_idx] = tokenizer._token_mask_id
                target_ids[mask_idx] = neg_id
            elif label == 1:
                source_ids[mask_idx] = tokenizer._token_mask_id
                target_ids[mask_idx] = pos_id
            batch_token_ids.append(source_ids)
            batch_segment_ids.append(segment_ids)
            batch_output_ids.append(target_ids)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_output_ids = sequence_padding(batch_output_ids)
                yield [
                    batch_token_ids, batch_segment_ids, batch_output_ids
                ], None
                batch_token_ids, batch_segment_ids, batch_output_ids = [], [], []


class CrossEntropy(Loss):
    """交叉熵作为loss，并mask掉输入部分
    """
    def compute_loss(self, inputs, mask=None):
        y_true, y_pred = inputs
        y_mask = K.cast(K.not_equal(y_true, 0), K.floatx())
        accuracy = keras.metrics.sparse_categorical_accuracy(y_true, y_pred)
        accuracy = K.sum(accuracy * y_mask) / K.sum(y_mask)
        self.add_metric(accuracy, name='accuracy')
        loss = K.sparse_categorical_crossentropy(y_true, y_pred)
        loss = K.sum(loss * y_mask) / K.sum(y_mask)
        return loss


# 加载预训练模型
model = build_transformer_model(
    config_path=config_path, checkpoint_path=checkpoint_path, with_mlm=True
)

# 训练用模型
y_in = keras.layers.Input(shape=(None,))
outputs = CrossEntropy(1)([y_in, model.output])

train_model = keras.models.Model(model.inputs + [y_in], outputs)
train_model.compile(optimizer=Adam(2e-5))
train_model.summary()

# 转换数据集
train_generator = data_generator(train_data, batch_size)
valid_generator = data_generator(valid_data, batch_size)


class Evaluator(keras.callbacks.Callback):
    def __init__(self):
        self.best_val_acc = 0.

    def on_epoch_end(self, epoch, logs=None):
        model.save_weights('mlm_model.weights')
        val_acc = evaluate(valid_generator)
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            model.save_weights('best_model.weights')
        print(
            u'val_acc: %.5f, best_val_acc: %.5f\n' %
            (val_acc, self.best_val_acc)
        )


def evaluate(data):
    total, right = 0., 0.
    for x_true, _ in data:
        x_true, y_true = x_true[:2], x_true[2]
        y_pred = model.predict(x_true)
        y_preds = []
        y_trues = []
        for i in range(len(y_true)):       # ⭐️ 记得修改和上面对应
            pred = y_pred[i, y_true[i].tolist().index(102)+3, [neg_id, pos_id]].argmax(axis=0)  # 选概率大的那一个字
            y_preds.append(pred)
            true = (y_true[i, y_true[i].tolist().index(102)+3] == pos_id).astype(int)  # pos的标签字为1，[0 0 1 1 0 1 0 01 0 0]
            y_trues.append(true)  
        total += len(y_trues)
        right += (np.array(y_trues) == np.array(y_preds)).sum()

    return right / total


if __name__ == '__main__':

    # few-shot
    # evaluator = Evaluator()
    # train_model.fit_generator(
    #     train_generator.forfit(),
    #     steps_per_epoch=len(train_generator),
    #     epochs=epochs,
    #     callbacks=[evaluator]
    # )

    # zero-shot
    val_acc = evaluate(valid_generator)
    print(val_acc)
else:

    model.load_weights('best_model.weights')
