#实现简单卷积神经网络对MNIST数据集进行分类：conv2d+active+pool+fc
import  csv
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

#设置算法超参数
learning_rate=0.001
train_epochs=1
batch_size=100
display_step=10

#Network Parmam
n_input=784
n_classes=10

#根据指定的维度返回初始化好的指定名称的权重
def WeightVariable(shape,name_str,stddev=0.1):
    initial=tf.random_normal(shape=shape,stddev=stddev,dtype=tf.float32)#标准正态分布初始化
   # initial = tf.truncated_normal(shape=shape, stddev=stddev, dtype=tf.float32)  # 切断正态分布初始化
    return tf.Variable(initial,name=name_str,dtype=tf.float32)
#根据指定的维度返回初始化好的指定名称的偏置
def BasisVariable(shape,name_str,stddev=0.0001):
    initial=tf.random_normal(shape=shape,stddev=stddev,dtype=tf.float32)
    #initial=tf.constant(stddev,shape=shape);
    return tf.Variable(initial,name=name_str,dtype=tf.float32)
#2维卷积层
def Conv2d(x,W,b,stride=1,padding='SAME'):
    with tf.name_scope('Wx_b'):
        y=tf.nn.conv2d(x,W,strides=[1,stride,stride,1],padding=padding)
        y=tf.nn.bias_add(y,b)
    return y
#非线性激活层
def activity(x,activity=tf.nn.relu,name='relu'):
    with tf.name_scope(name):
        y=activity(x)
    return y
#2维卷积池化层
def pool2d(x,pool=tf.nn.max_pool,k=2,stride=2):
    return pool(x,ksize=[1,k,k,1],strides=[1,stride,stride,1],padding='VALID')
#全连接层
def FullConnection(x,W,b,active=tf.nn.relu,act_name='relu'):
    with tf.name_scope('Wx_b'):
        y=tf.matmul(x,W)
        y=tf.add(y,b)
    with tf.name_scope(act_name):
        y=active(y)
    return y
#通用的评估函数，用来评估模型在给定的数据集上的损失和准确率
def EvaluatedModeOnDataset(sess,images,labels):
    n_samples=images.shape[0]
    per_batch_size=100
    loss=0;
    acc=0
    if(n_samples<=per_batch_size):
        batch_count=1
        loss,acc=sess.run([cross_entrcopy_loss,accuray],
                          feed_dict={X_origin:images,Y_true:labels,learning_rate_train:learning_rate })
    else:
        batch_count=int(n_samples/per_batch_size)
        batch_start=0
        for idx in range(batch_count):
            batch_loss,batch_acc=sess.run([cross_entrcopy_loss,accuray],
                                          feed_dict={X_origin:images[batch_start:batch_start+per_batch_size,:],
                                                     Y_true:labels[batch_start:batch_start+per_batch_size,:],
                                                     learning_rate_train:learning_rate})
            batch_start+=per_batch_size
            loss+=batch_loss
            acc+=batch_acc
        return loss/batch_count,acc/batch_count
#调用上面的函数构造计算图
with tf.Graph().as_default():
    #计算图输入
    with tf.name_scope('Inputs'):
        X_origin=tf.placeholder(tf.float32,[None,n_input],name='X_origin')
        Y_true=tf.placeholder(tf.float32,[None,n_classes],name='Y_true')
        #把N*784维的张量转换成N*28*28*1的张量
        X_image=tf.reshape(X_origin,[-1,28,28,1])
    #计算图前向推断过程
    with tf.name_scope('inference'):
        #第一卷积层
        with tf.name_scope('Conv2d'):
            conv_kenerl_num=16
            weights=WeightVariable(shape=[5,5,1,conv_kenerl_num],name_str='weights')
            basis=BasisVariable(shape=[conv_kenerl_num],name_str='basis')
            conv_out=Conv2d(X_image,weights,basis,stride=1,padding='VALID')
        #非线性激活层
        with tf.name_scope('activity'):
            activaty_out=activity(conv_out,activity=tf.nn.relu,name='relu')
        #池化层
        with tf.name_scope('pool2d'):
           pool_out=pool2d(activaty_out,pool=tf.nn.max_pool,k=2,stride=2)
        #将二维卷积转换为一维向量
        with tf.name_scope('Feature'):
            feature_out=tf.reshape(pool_out,[-1,12*12*conv_kenerl_num])
        #全连接层
        with tf.name_scope('FC_Linear'):
            weights=WeightVariable([12*12*conv_kenerl_num,n_classes],name_str='weights')
            basis=BasisVariable([n_classes],name_str='basis')
            Yprep_logists=FullConnection(feature_out,weights,basis,active=tf.identity,act_name='identity')
    #定义损失层
    with tf.name_scope('Loss'):
        cross_entrcopy_loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y_true,logits=Yprep_logists))
    #定义训练层
    with tf.name_scope('Train'):
        learning_rate_train=tf.placeholder(tf.float32)
        optiminer=tf.train.AdamOptimizer(learning_rate=learning_rate_train)
        trainer=optiminer.minimize(cross_entrcopy_loss)
    #定义评估模型
    with tf.name_scope('Evaluate'):
        correct_prep=tf.equal(tf.argmax(Y_true,1),tf.argmax(Yprep_logists,1))
        accuray=tf.reduce_mean(tf.cast(correct_prep,tf.float32))
    #t添加所有初始化节点
    initVariable=tf.global_variables_initializer()

    print('把计算图写入文件，在TensorBoard中查看')
    write=tf.summary.FileWriter(logdir='logs/simpleConv2d',graph=tf.get_default_graph())
    write.close()

    #加载数据集
    mnist=input_data.read_data_sets('mnist_data/',one_hot=True)
    #将评估结果保存到文件
    result_list=list()
    #写入参数配置
    result_list.append(['learning_rate',learning_rate,'tran_epochs',train_epochs,'bactch_size',batch_size,'display_step',display_step])
    result_list.append(['train_step','train_loss','validation_loss','train_step','train_accuracy','validation_accuracy'])
   #启动计算图
    with tf.Session() as sess:
        sess.run(initVariable)
        total_batches=int(mnist.train.num_examples/batch_size)
        print('Per batch_size',batch_size)
        print('Train exmaple count',mnist.train.num_examples)
        print('Train total count',total_batches)
        #train_step
        train_steps=0
        for epoch in range(train_epochs):
            for batch_index in range(total_batches):
                batch_x,batch_y=mnist.train.next_batch(batch_size)
                #运行优化器训练节点
                sess.run(trainer,feed_dict={X_origin:batch_x,Y_true:batch_y,learning_rate_train:learning_rate})
                train_steps+=1
                #每训练display_step次，计算当前模型的损失和分类准确率
                if train_steps % display_step==0:
                    #计算当前模型在目前（最近）见过的display_step 个batchsize的训练集上的损失和分类准确率
                    start_indx=max(0,(batch_index-display_step)*batch_size)
                    end_indx=batch_index*batch_size
                    train_loss,train_acc=EvaluatedModeOnDataset(sess,
                                                                mnist.train.images[start_indx:end_indx,:],
                                                                mnist.train.labels[start_indx:end_indx,:])
                    print('Training Step:'+str(train_steps)+
                          ', Training Loss='+'{:.6f}'.format(train_loss)+
                          ', Training Accuracy='+'{:.5f}'.format(train_acc))
                    #计算当前模型在验证集上的损失和分类准确率
                    validation_loss,validation_acc=EvaluatedModeOnDataset(sess,
                                                                           mnist.validation.images,
                                                                           mnist.validation.labels)
                    print('Training Step:'+str(train_steps)+
                          ', Validation Loss='+'{:.6f}'.format(validation_loss)+
                          ', Validation Accuracy='+'{:.5f}'.format(validation_acc))
                    #将评估结果保存到文件中
                    result_list.append([train_steps,train_loss,validation_loss,train_steps,train_acc,validation_acc])

        print('训练完毕')
        #计算指定数量的测试集上的准确率
        test_sample_count=mnist.test.num_examples
        test_loss,test_acc=EvaluatedModeOnDataset(sess,mnist.test.images,mnist.test.labels)
        print('Test sample count:',test_sample_count)
        print('Test Loss',test_loss)
        print('Test Acc',test_acc)
        result_list.append(['teststep','loss',test_loss,'accuray',test_acc])

        #将评价结果保存到文件
        result_file=open('evluate_results.csv','w',newline='')
        csv_write=csv.writer(result_file,dialect='excel')
        for row in result_list:
            csv_write.writerow(row)























