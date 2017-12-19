# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 10:32:07 2017

@author: PALM
"""

import tensorflow as tf  
import numpy as np  
import datetime,sys  
from tensorflow.contrib import learn  
  
training_epochs = 3  
train_num = 4  
# 运行Graph  
with tf.Session() as sess:  
  
    #定义模型  
    BATCH_SIZE = 2  
    # 构建训练数据输入的队列  
    # 生成一个先入先出队列和一个QueueRunner,生成文件名队列  
    filenames = ['a.csv']  
    filename_queue = tf.train.string_input_producer(filenames, shuffle=True)  
    # 定义Reader  
    reader = tf.TextLineReader()  
    key, value = reader.read(filename_queue)  
    # 定义Decoder  
    # 编码后的数据字段有24,其中22维是特征字段,2维是lable字段,label是二分类经过one-hot编码后的字段  
    #更改了特征,使用不同的解析参数  
    record_defaults = [[1]]*5  
    col1,col2,col3,col4,col5 = tf.decode_csv(value,record_defaults=record_defaults)  
    features = tf.stack([col1,col2,col3,col4])  
    label = tf.stack([col5])  
  
    example_batch, label_batch = tf.train.shuffle_batch([features,label], batch_size=BATCH_SIZE, capacity=20000, min_after_dequeue=4000, num_threads=2)  
  
    sess.run(tf.global_variables_initializer())  
    coord = tf.train.Coordinator()#创建一个协调器，管理线程  
    threads = tf.train.start_queue_runners(coord=coord)#启动QueueRunner, 此时文件名队列已经进队。  
    #开始一个epoch的训练  
    for epoch in range(training_epochs):  
        total_batch = int(train_num/BATCH_SIZE)  
        #开始一个epoch的训练  
        for i in range(total_batch):  
            X,Y = sess.run([example_batch, label_batch])  
            print(X,':',Y ) 
    coord.request_stop()  
    coord.join(threads)  