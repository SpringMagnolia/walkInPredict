# coding=utf-8

import tensorflow as tf
import random
import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer


def neural_net_model():  # 建立模型
    # 准备数据
    x_data = tf.placeholder(shape=[None, 37], dtype=tf.float32)
    y_true = tf.placeholder(shape=[None, 1], dtype=tf.float32)
    # layer1
    W_1 = tf.Variable(tf.random_uniform([37, 10]))
    b_1 = tf.Variable(tf.zeros([10]))
    layer_1 = tf.add(tf.matmul(x_data, W_1), b_1)
    layer_1 = tf.nn.relu(layer_1)
    # laryer2
    W_2 = tf.Variable(tf.random_uniform([10, 10]))
    b_2 = tf.Variable(tf.zeros([10]))
    layer_2 = tf.add(tf.matmul(layer_1, W_2), b_2)
    layer_2 = tf.nn.relu(layer_2)
    # layer-fc
    W_O = tf.Variable(tf.random_uniform([10, 1]))
    b_O = tf.Variable(tf.zeros([1]))
    y_predict = tf.add(tf.matmul(layer_2, W_O), b_O)
    loss = tf.reduce_mean(tf.square(y_true - y_predict))
    train_op = tf.train.AdamOptimizer(0.001).minimize(loss)
    return x_data, y_true, y_predict, loss, train_op


def get_total_data():
    df = pd.read_csv('./train_data.csv')
    # 转化为str之后进行onehot操作
    for i in ["device_angle", "distance_to_door", "AM_or_PM", "mall_or_street", "average_person_size"]:
        df[i] = df[i].apply(lambda x: str(x))
    dict_data = df.to_dict(orient="records")
    dict_vec = DictVectorizer()
    # 进行特征提取
    df_data = dict_vec.fit_transform(dict_data).toarray()  # [none,37]
    return df_data


def get_next_batch(df_data, batch_size=50):  # 获取数据
    start_rownum = random.randint(0, df_data.shape[0] - 1)

    if start_rownum + batch_size > df_data.shape[0]:
        temp_data = np.vstack((df_data, df_data))
    else:
        temp_data = df_data
    chosed_data = temp_data[start_rownum:start_rownum + batch_size]
    data = tf.constant(chosed_data[:, :-1], dtype=tf.float32).eval()
    target = tf.constant(chosed_data[:, [-1]], dtype=tf.float32).eval()
    return data, target


# 4. 执行
def main(argv):
    df_data = get_total_data()
    x_data, y_true, y_predict, loss, train_op = neural_net_model()
    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        # 实例化saver
        saver = tf.train.Saver()
        checkpoint = tf.train.latest_checkpoint("./")
        if checkpoint:
            saver.restore(sess, checkpoint)

        for i in range(20000):
            data_batch, lable_bath = get_next_batch(df_data, batch_size=500)
            _, _loss = sess.run([train_op, loss], feed_dict={x_data: data_batch, y_true: lable_bath})
            print("i:", i + 1, "loss", _loss)

            if (i + 1) % 100 == 0:
                saver.save(sess, "./sw_relu_batch.ckpt")


if __name__ == '__main__':
    tf.app.run()
