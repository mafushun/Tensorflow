import numpy as np
import tensorflow as tf

#构建计算图
with tf.Graph().as_default():
    with tf.name_scope('Input'):
        #占位符
        X=tf.placeholder(tf.float32,shape=[None,784],name='X')
        Y_true=tf.placeholder(tf.float32,shape=[None,10],name='Y_true')
    with tf.name_scope('Inferenc'):
        #参数
        W=tf.Variable(tf.zeros([784,10]),name='weight')
        b=tf.Variable(tf.zeros([10]),name='baise')
        logist=tf.add(tf.matmul(X,W),b)
        with tf.name_scope('softmax'):
            Y_pre = tf.nn.softmax(logits=logist)
    with tf.name_scope('TrainLoss'):
        TrainLoss=tf.reduce_mean(-tf.reduce_sum(Y_true*tf.log(Y_pre),axis=1))
    with tf.name_scope('Train'):
        operimater=tf.train.GradientDescentOptimizer(0.01)
        Train=operimater.minimize(TrainLoss)

    init=tf.global_variables_initializer()

    write=tf.summary.FileWriter(logdir='logs/',graph=tf.get_default_graph())
    write.close()

