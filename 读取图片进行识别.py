import tensorflow as tf
import os
import numpy as np

import pandas as pd
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

data = pd.read_csv("D://MLDatabase//db//FER2013//fer2013test.csv", dtype='a')
img_data = np.array(data['pixels'])
label=np.array(data['emotion'])


# ����shape���岢��ʼ��bias
def bias_variable(shape):
    initial=tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

# ���ݸ���shape���岢��ʼ������˵�Ȩֵ����
def weight_variable(shape):
    # ��׼��Ϊ0.1
    initial=tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

def conv2d(x,W):
    # ����һ������Ϊx,Ȩ��ΪW������ά�ȵĲ�������1��û�б�Ե�����ľ����
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='VALID')

def max_pool_2x2(x):
    # ����ػ���
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')



keep_prob = tf.placeholder("float")

def model(x):
    W_conv1 = weight_variable([3, 3, 1, 16])
    b_conv1 = bias_variable([16])

    h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = weight_variable([2, 2, 16, 32])
    b_conv2 = bias_variable([32])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    W_fc1 = weight_variable([11 * 11 * 32, 500])
    b_fc1 = bias_variable([500])

    h_pool3_flat = tf.reshape(h_pool2, [-1, 11 * 11 * 32])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)

    W_fc2 = weight_variable([500, 500])
    b_fc2 = bias_variable([500])

    h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)
    h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

    W_fc3 = weight_variable([500, 7])
    b_fc3 = bias_variable([7])

    y_conv = tf.nn.softmax(tf.matmul(h_fc2_drop, W_fc3) + b_fc3)

    return y_conv

traindata=[]
for im in img_data:
    td1=[]
    for s in im.strip().split(' '):
        td1.append(int(s))
    traindata.append(np.array(td1).reshape(48,48))
trainlabel=[]

for l in label:
    tl1=np.zeros(7)
    tl1[int(l)]=1
    trainlabel.append(tl1)

def pre():
    X=tf.placeholder("float", shape=[None, 48,48, 1])

    module_file = tf.train.latest_checkpoint('./model/')
    test=model(X)
    with tf.Session() as sess:
        predict=[]
        tf.train.Saver().restore(sess, module_file)
        print(module_file)
        for i in range(1):
            p=sess.run(test,feed_dict={X:[traindata[i].reshape((48,48,1))],keep_prob:0.5})
pre()
