import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 进行数据读取
data = pd.read_csv("D://MLDatabase//db//FER2013//fer2013.csv", dtype='a')
label = np.array(data['emotion'])
img_data = np.array(data['pixels'])

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

# 根据shape定义并初始化bias
def bias_variable(shape):
    initial=tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

# 根据给定shape定义并初始化卷积核的权值变量
def weight_variable(shape):
    # 标准差为0.1
    initial=tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

def conv2d(x,W):
    # 定义一个输入为x,权重为W，各个维度的步长都是1的没有边缘补偿的卷积层
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='VALID')

def max_pool_2x2(x):
    # 定义池化层
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

x = tf.placeholder("float", shape=[None, 48,48, 1])
y_ = tf.placeholder("float", shape=[None,7])
keep_prob = tf.placeholder("float")

def model():
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
    rmse = tf.sqrt(tf.reduce_mean(tf.square(y_ - y_conv)))
    return y_conv, rmse

VALIDATION_SIZE = 100    #验证集大小
EPOCHS = 100             #迭代次数
BATCH_SIZE = 32          #每个batch大小，稍微大一点的batch会更稳定
EARLY_STOP_PATIENCE = 10 #控制early stopping的参数


def input_data(test=False):
    X=np.array(traindata).reshape((-1,48,48,1))/255
    y=np.array(trainlabel)
    return X, y

def save_model(saver,sess,save_path):
        path = saver.save(sess, save_path)
        print ('model save in :{0}'.format(path))

if __name__ == '__main__':
    sess = tf.InteractiveSession()
    y_conv, rmse = model()
    train_step = tf.train.AdamOptimizer(1e-3).minimize(rmse)

    #变量都要初始化
    sess.run(tf.global_variables_initializer())
    X,y = input_data()
    X_valid, y_valid = X[:VALIDATION_SIZE], y[:VALIDATION_SIZE]
    X_train, y_train = X[VALIDATION_SIZE:], y[VALIDATION_SIZE:]

    best_validation_loss = 1000000.0
    current_epoch = 0
    TRAIN_SIZE =6432
    train_index = list(range(TRAIN_SIZE))
    np.random.shuffle(train_index)
    X_train, y_train = X_train[train_index], y_train[train_index]

    saver = tf.train.Saver()

    print ('begin training..., train dataset size:{0}'.format(TRAIN_SIZE))
    for i in range(1):
        np.random.shuffle(train_index)  #每个epoch都shuffle一下效果更好
        X_train, y_train = X_train[train_index], y_train[train_index]

        for j in range(0,TRAIN_SIZE,BATCH_SIZE):
            print ('epoch {0}, train {1} samples done...'.format(i,j))

            train_step.run(feed_dict={x:X_train[j:j+BATCH_SIZE],
                y_:y_train[j:j+BATCH_SIZE], keep_prob:0.5})


        train_loss = rmse.eval(feed_dict={x:X_train, y_:y_train, keep_prob: 1.0})
        validation_loss = rmse.eval(feed_dict={x:X_valid, y_:y_valid, keep_prob: 1.0})


        print ('epoch {0} done! validation loss:{1}'.format(i, validation_loss*64.0))
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            current_epoch = i
            save_model(saver,sess,"C:/Users/14229/PycharmProjects/untitled4/model/my.tfmodel")
               #即时保存最好的结果
        elif (i - current_epoch) >= EARLY_STOP_PATIENCE:
            print('early stopping')
            break

