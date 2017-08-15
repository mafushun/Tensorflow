import csv
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
#定义参数
learning_rate_init=0.01
train_epoches=1
batch_size=100
display_step=10

#NetWork Parma
n_input=784
n_output=10

#定义函数 初始化权重，偏置，卷积层（conv2d）,激活函数（activation）,池化层（pool），全链接成（FC）
#定义权重
def WeightVariable(shape,name,stddev=0.1):
    #标准正太分布初始化
    initial=tf.random_normal(shape,stddev=stddev)
    return tf.Variable(initial,dtype=tf.float32,name=name)
#初定义偏置函数
def BasisVariable(shape,name,sttdev=0.0001):
    initial=tf.random_normal(shape=shape,stddev=sttdev,dtype=tf.float32)
    return tf.Variable(initial,name=name,dtype=tf.float32)
#定义卷积层
def Conv2d(x,w,b,stride=1,padding='SAME'):
    with tf.name_scope('wx_b'):
        y=tf.nn.conv2d(x,w,strides=[1,stride,stride,1],padding=padding)
        y=tf.nn.bias_add(y,b)
    return y
#定义激活函数
def Activation(x,activiation=tf.nn.relu,name='relu'):
    with tf.name_scope(name):
        y=activiation(x)
    return y
#定义池化层
def pool(x,pool=tf.nn.max_pool,k=2,stride=2):
    return pool(x, ksize=[1, k, k, 1], strides=[1, stride, stride, 1], padding='VALID')
#二维数据转换成一维数据
def FeatureShape(x,reshapes):
    return tf.reshape(x,reshapes)
def FC_linear(x,w,b,activation=tf.nn.relu,act_name='relu'):
    with tf.name_scope('Wx_b'):
        y=tf.matmul(x,w)
        y=tf.add(y,b)
    with tf.name_scope(act_name):
        y=activation(y)
    return y

#通用的评估函数，用来评估模型在给定的数据集上的损失和准确率
def EvaluatedModeOnDataset(sess,images,labels):
    n_samples=images.shape[0]
    per_batch_size=100
    loss=0;
    acc=0
    if(n_samples<=per_batch_size):
        batch_count=1
        loss,acc=sess.run([loss_out,accuary],
                          feed_dict={X_orgin:images,Y_true:labels,learning_rates:learning_rate_init })
    else:
        batch_count=int(n_samples/per_batch_size)
        batch_start=0
        for idx in range(batch_count):
            batch_loss,batch_acc=sess.run([loss_out,accuary],
                                          feed_dict={X_orgin:images[batch_start:batch_start+per_batch_size,:],
                                                     Y_true:labels[batch_start:batch_start+per_batch_size,:],
                                                     learning_rates:learning_rate_init})
            batch_start+=per_batch_size
            loss+=batch_loss
            acc+=batch_acc
        return loss/batch_count,acc/batch_count

#构建计算图
with tf.Graph().as_default():
    #输入数据 占位符
    with tf.name_scope("input"):
      X_orgin=tf.placeholder(tf.float32,[None,n_input],name='X_orgin')
      Y_true=tf.placeholder(tf.float32,[None,n_output],name='Y_true')
      X_images=tf.reshape(X_orgin,[-1,28,28,1],name='X_images')
    with tf.name_scope('Inference'):
        #卷积层
        with tf.name_scope('Conv2d'):
            weight=WeightVariable([5,5,1,16],name='weight')
            basis=BasisVariable([16],name='basis')
            conv2d_out=Conv2d(X_images,weight,basis)
        #激活层
        with tf.name_scope('Activation'):
            activation_out=Activation(conv2d_out,activiation=tf.nn.relu,name='relu')
        #池化层
        with tf.name_scope('pool2d'):
            pool_out=pool(activation_out,pool=tf.nn.max_pool,k=2,stride=2)
        #全链接层
        with tf.name_scope('FC_linear'):
            w=WeightVariable([12*12*26,200],name='weight')
            b=BasisVariable([200],name='basis')
            fc_out=FC_linear(FeatureShape(pool_out,[12*12*26]),w,b,activation=tf.nn.relu,act_name='relue')
    #定义损失
    with tf.name_scope('Loss'):
        loss_out=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(Y_true,fc_out))
    #采用tf提供的优化器训练模型，计算梯度，反向传播更新参数
    with tf.name_scope('Train'):
        learning_rates=tf.placeholder(tf.float32)
        optimier=tf.train.AdamOptimizer(learning_rate=learning_rates)
        train_out=optimier.minimize(loss_out)
    #评估
    with  tf.name_scope('Evlation'):
        correct_de=tf.equal(tf.argmax(Y_true,1),tf.argmax(fc_out,1))
        accuary=tf.reduce_mean(tf.cast(correct_de,tf.float32))

    #初始化
    init=tf.global_variables_initializer()

    #把计算图训练写入文件，供tensorboard 查看
    write=tf.summary.FileWriter('logs/simpleConv2_k',graph=tf.get_default_graph())
    write.close()

    #加载数据
    mnist=input_data.read_data_sets('mnist_data',one_hot=True);
    # 将评估结果保存到文件
    result_list = list()
    # 写入参数配置
    result_list.append(
        ['learning_rate', learning_rates, 'tran_epochs', train_epoches, 'bactch_size', batch_size, 'display_step',
         display_step])
    result_list.append(
        ['train_step', 'train_loss', 'validation_loss', 'train_step', 'train_accuracy', 'validation_accuracy'])

    #开始计算
    with tf.Session() as sess:
        sess.run(init)
        batch_count=init(mnist.train.num_examples/batch_size)
        print('Per batch_size',batch_size)
        print('Train exmaple count',mnist.train.num_examples)
        print('Train total count',batch_count)
        train_steps = 0
        for epoche in  range(train_epoches):
            for batch_indx in range(batch_count):
                batch_x,batch_y=mnist.train.next_batch(batch_size)
                #运行优化器节点
                sess.run(train_out,feed_dict={X_orgin:batch_x,Y_true:batch_y,learning_rates:learning_rate_init})
                train_steps += 1
                # 每训练display_step次，计算当前模型的损失和分类准确率
                if train_steps % display_step == 0:
                    # 计算当前模型在目前（最近）见过的display_step 个batchsize的训练集上的损失和分类准确率
                    start_indx = max(0, (batch_indx - display_step) * batch_size)
                    end_indx = batch_indx * batch_size
                    train_loss, train_acc = EvaluatedModeOnDataset(sess,
                                                                   mnist.train.images[start_indx:end_indx, :],
                                                                   mnist.train.labels[start_indx:end_indx, :])
                    print('Training Step:' + str(train_steps) +
                          ', Training Loss=' + '{:.6f}'.format(train_loss) +
                          ', Training Accuracy=' + '{:.5f}'.format(train_acc))
                    # 计算当前模型在验证集上的损失和分类准确率
                    validation_loss, validation_acc = EvaluatedModeOnDataset(sess,
                                                                             mnist.validation.images,
                                                                             mnist.validation.labels)
                    print('Training Step:' + str(train_steps) +
                          ', Validation Loss=' + '{:.6f}'.format(validation_loss) +
                          ', Validation Accuracy=' + '{:.5f}'.format(validation_acc))
                    # 将评估结果保存到文件中
                    result_list.append(
                        [train_steps, train_loss, validation_loss, train_steps, train_acc, validation_acc])

            print('训练完毕')
            # 计算指定数量的测试集上的准确率
            test_sample_count = mnist.test.num_examples
            test_loss, test_acc = EvaluatedModeOnDataset(sess, mnist.test.images, mnist.test.labels)
            print('Test sample count:', test_sample_count)
            print('Test Loss', test_loss)
            print('Test Acc', test_acc)
            result_list.append(['teststep', 'loss', test_loss, 'accuray', test_acc])

            # 将评价结果保存到文件
            result_file = open('evluate_results.csv', 'w', newline='')
            csv_write = csv.writer(result_file, dialect='excel')
            for row in result_list:
                csv_write.writerow(row)











