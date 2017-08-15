import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# 产生训练数据集
train_X = np.asarray([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,
                         7.042,10.791,5.313,7.997,5.654,9.27,3.1])
train_Y = np.asarray([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,
                         2.827,3.465,1.65,2.904,2.42,2.94,1.3])
n_train_samples = train_X.shape[0]
print('训练样本数量: ', n_train_samples)
# 产生测试样本
test_X = np.asarray([6.83, 4.668, 8.9, 7.91, 5.7, 8.7, 3.1, 2.1])
test_Y = np.asarray([1.84, 2.273, 3.2, 2.831, 2.92, 3.24, 1.35, 1.03])
n_test_samples = test_X.shape[0]
print('测试样本数量: ', n_test_samples)

# 展示原始数据分布
plt.plot(train_X,train_Y,'ro',label='Original Train Points')
plt.plot(test_X,test_Y,'b*',label='Original Test Points')
plt.legend()
plt.show()


#计算图
with tf.Graph().as_default():
    #输入X和真实值Y 占位符
    with tf.name_scope('Input'):
        X=tf.placeholder(tf.float32,name='X')
        Y_true=tf.placeholder(tf.float32,name='Y_true')

    with tf.name_scope('inference'):
        # 模型参数
        W=tf.Variable(tf.zeros([1]),name='Weight')
        b=tf.Variable(tf.zeros([1]),name='Biase');
        #inference: y=wx+b
        Y_pred=tf.add(tf.multiply(X,W),b)
    with tf.name_scope('Loss'):
        #添加损失
        TrainLoss=tf.reduce_mean(tf.pow((Y_true-Y_pred),2))/2
    with tf.name_scope('Train'):
        Optimizer=tf.train.GradientDescentOptimizer(0.01)
        TrainOpt=Optimizer.minimize(TrainLoss)
    with tf.name_scope('Evalate'):
        #添加评估节点
        EvaLoss=tf.reduce_mean(tf.pow((Y_true-Y_pred),2))/2
    #Initial:添加所有Variable类型的变量的初始化节点
    InitOp = tf.global_variables_initializer();
    #保存计算图
    writer=tf.summary.FileWriter(logdir='logs/',graph=tf.get_default_graph())
    writer.close()
    print('启动会话，开启训练评估模式，让计算图跑起来')
    sess=tf.Session()
    sess.run(InitOp)

    print("不断的迭代训练并测试模型")
    for step in range(50000):
        _,train_loss,train_w,train_b=sess.run([TrainOpt,TrainLoss,W,b],feed_dict={X:train_X,Y_true:train_Y})
        # 每隔几步训练完之后输出当前模型的损失
        if (step + 1) % 5 == 0:
            print("Step:", '%04d' % (step + 1), "train_loss=", "{:.9f}".format(train_loss),
                  "W=", train_w, "b=", train_b)
            # 每隔几步训练完之后对当前模型进行测试
        if (step + 1) % 5 == 0:
            test_loss, test_w, test_b = sess.run([EvaLoss, W, b],feed_dict={X: test_X, Y_true: test_Y})
            print("Step:", '%04d' % (step + 1), "test_loss=", "{:.9f}".format(test_loss),
                      "W=", test_w, "b=", test_b)
    print('训练完毕！')
    W,b=sess.run([W,b])
    print("得到的模型参数：", "W=", W, "b=", b, )
    # 展示拟合曲线
    plt.plot(train_X, train_Y, 'ro', label='Original Train Points')
    plt.plot(test_X, test_Y, 'b*', label='Original Test Points')
    plt.plot(train_X, W * train_X + b, label='Fitted Line')
    plt.legend()
    plt.show()




















