# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 00:03:59 2020

@author: Dell
"""


# change the structure of Generator and Discriminator


import tensorflow as tf
import datetime

# Load data

import numpy as np
import matplotlib.pyplot as plt
import random
import math
import pandas as pd

mea1= pd.read_csv ('c_gamf9064_1000_1.csv',header=None)##9064 in matlab > 64*90 in python
mea2= pd.read_csv ('c_gamf9064_1000_2.csv',header=None)#10000 measurments perfile
mea3= pd.read_csv ('c_gamf9064_1000_3.csv',header=None)
mea4= pd.read_csv ('c_gamf9064_1000_4.csv',header=None)
mea5= pd.read_csv ('c_gamf9064_1000_5.csv',header=None)


mea6=pd.concat([mea1,mea2,mea3,mea4,mea5])

mea6=np.array(mea6).astype(np.float32)
# lump1= pd.read_csv ('D:\GitHub\LSOM\obf_10000.csv',header=None)
# lump2=np.array(lump1).astype(np.float32)

H= pd.read_csv ('Hmatrix9064.csv',header=None)
H= np.array(H).astype(np.float32)



# F= pd.read_csv ('D:\GitHub\LSOM\F.csv',header=None)
# F= np.array(F).astype(np.float32)
# F=np.transpose(F)

# G=np.matmul(H,lump2[1,]).astype(np.float32)
# Is=np.transpose(np.reshape(G,[60,64]).astype(np.float32))
# plt.imshow(Is,cmap='Greys',interpolation=None)


# GG=np.matmul(H,np.transpose(np.reshape(lump2[1,], [-1, 64*64])))
# Iss=np.reshape(GG,[60,64])
# plt.imshow(Iss,cmap='Greys',interpolation=None)

# Is=mea2[1,].reshape(64,60)
# plt.imshow(Is,cmap='Greys',interpolation=None)

# Im=lump2[1,].reshape(64,64)
# Im =np.transpose(Im)
# plt.imshow(Im,cmap='Greys',interpolation=None)



# for j in range(1,2021):
    
#     filename= "D:\GitHub\GAN project\lump\{}th.jpg".format(j)
#     if j % 100== 0 or j == 1:
#         print(filename)
#     pil_im = Image.open(filename).convert('L')
# #pil_im.show()
#     I_np=np.array(pil_im)
#     I_nps=I_np[47:562,195:710]
#     Is=np.zeros((32,32))
#     for i in range(0,32):
#         for k in range(0,32):
#             Is[i,k]=I_nps[15*i,15*k]
#     Is=256-Is
#     plt.imshow(Is,cmap='Greys',interpolation=None)
#     lump[j-1]=Is.reshape(1,1024)
    

class lumpypic:
    
    def __init__(self,data):
        self.data=data
        self.n=len(data)
    def next_batch(self,batchsize):
        self.batchdata=np.empty((batchsize,64*90))
        for i in range(0,batchsize):
            index=math.floor(random.uniform(0,49999))
            self.batchdata[i]=self.data[index]
        return self.batchdata
    

input=lumpypic(mea6)
x=input.next_batch(10)


# Define the discriminator network
def discriminator(images, reuse_variables=None):
    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variables) as scope:
        d_w1 = tf.get_variable('d_w1', [3, 3, 1, 32], initializer=tf.truncated_normal_initializer(stddev=0.02))
        d_b1 = tf.get_variable('d_b1', [32], initializer=tf.constant_initializer(0))
        d1 = tf.nn.conv2d(input=images, filter=d_w1, strides=[1, 1, 1, 1], padding='SAME')
        d1 = d1 + d_b1
        d1 = tf.nn.relu(d1)
        d1 = tf.nn.max_pool(d1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # Second convolutional and pool layers
        # This finds 64 different 3 x 3 pixel features
        d_w2 = tf.get_variable('d_w2', [3, 3, 32, 64], initializer=tf.truncated_normal_initializer(stddev=0.02))
        d_b2 = tf.get_variable('d_b2', [64], initializer=tf.constant_initializer(0))
        d2 = tf.nn.conv2d(input=d1, filter=d_w2, strides=[1, 1, 1, 1], padding='SAME')
        d2 = d2 + d_b2
        d2 = tf.nn.relu(d2)
        d2 = tf.nn.max_pool(d2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        
         #third convolutional and pool layers
        # This finds 128 different 3x3 pixel features
        d_w3 = tf.get_variable('d_w3', [3, 3, 64, 128], initializer=tf.truncated_normal_initializer(stddev=0.02))
        d_b3 = tf.get_variable('d_b3', [128], initializer=tf.constant_initializer(0))
        d3 = tf.nn.conv2d(input=d2, filter=d_w3, strides=[1, 1, 1, 1], padding='SAME')
        d3 = d3 + d_b3
        d3 = tf.nn.relu(d3)
        d3 = tf.nn.max_pool(d3, ksize=[1, 2,2, 1], strides=[1, 2, 2, 1], padding='SAME')
        
        #fourth convolutional and pool layers
        # This finds 256 different 3 x 3 pixel features
        d_w4 = tf.get_variable('d_w4', [3, 3, 128, 256], initializer=tf.truncated_normal_initializer(stddev=0.02))
        d_b4 = tf.get_variable('d_b4', [256], initializer=tf.constant_initializer(0))
        d4 = tf.nn.conv2d(input=d3, filter=d_w4, strides=[1, 1, 1, 1], padding='SAME')
        d4 = d4 + d_b4
        d4 = tf.nn.relu(d4)
        d4 = tf.nn.max_pool(d4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # First fully connected layer
        d_w5= tf.get_variable('d_w5', [4* 6 * 256, 2048], initializer=tf.truncated_normal_initializer(stddev=0.02))
        d_b5 = tf.get_variable('d_b5', [2048], initializer=tf.constant_initializer(0))
        d5 = tf.reshape(d4, [-1,4* 6* 256])
        d5= tf.matmul(d5, d_w5)
        d5 = d5+ d_b5
        d5= tf.nn.relu(d5)

        # Second fully connected layer
        d_w6 = tf.get_variable('d_w6', [2048, 1], initializer=tf.truncated_normal_initializer(stddev=0.02))
        d_b6 = tf.get_variable('d_b6', [1], initializer=tf.constant_initializer(0))
        d6 = tf.matmul(d5, d_w6) + d_b6
        d6= tf.math.sigmoid(d6)
        
        return d6

# Define the generator network
def generator(batch_size, z_dim):
    z = tf.random_normal([batch_size, z_dim], mean=0, stddev=1, name='z')
    g_w1 = tf.get_variable('g_w1', [z_dim, 64*64*4], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
    g_b1 = tf.get_variable('g_b1', [32*32*16], initializer=tf.truncated_normal_initializer(stddev=0.02))
    g1 = tf.matmul(z, g_w1) + g_b1
    g1 = tf.reshape(g1, [-1, 64, 64, 4])
    g1 = tf.contrib.layers.batch_norm(g1, epsilon=1e-5, scope='g_b1')
    g1 = tf.nn.leaky_relu(g1)

    # Generate 50 features
    g_w2 = tf.get_variable('g_w2', [3, 3, 4, z_dim/2], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
    g_b2 = tf.get_variable('g_b2', [z_dim/2], initializer=tf.truncated_normal_initializer(stddev=0.02))
    g2 = tf.nn.conv2d(input=g1, filter=g_w2, strides=[1, 2, 2, 1], padding='SAME')
    g2 = g2 + g_b2
    g2 = tf.contrib.layers.batch_norm(g2, epsilon=1e-5, scope='g_b2')
    g2 = tf.nn.leaky_relu(g2)



    # Generate 25 features
    g_w3 = tf.get_variable('g_w3', [3, 3, z_dim/2, z_dim/4], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
    g_b3 = tf.get_variable('g_b3', [z_dim/4], initializer=tf.truncated_normal_initializer(stddev=0.02))
    g3 = tf.nn.conv2d(g2, g_w3, strides=[1, 2, 2, 1], padding='SAME')
    g3 = g3 + g_b3
    g3 = tf.contrib.layers.batch_norm(g3, epsilon=1e-5, scope='g_b3')
    g3 = tf.nn.leaky_relu(g3)
    #up sampling to 4 times the shape
    g3=tf.keras.layers.UpSampling2D(size=(4, 4))(g3)


 

    # thrid convolution 
    g_w4 = tf.get_variable('g_w4', [2, 2, z_dim/4, 1], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
    g_b4 = tf.get_variable('g_b4', [1], initializer=tf.truncated_normal_initializer(stddev=0.02))
    g4 = tf.nn.conv2d(g3, g_w4, strides=[1, 1, 1, 1], padding='SAME')
    g4 = g4 + g_b4
    g4 = tf.nn.leaky_relu(g4)
    # thrid convolution 
    g_w5 = tf.get_variable('g_w5', [2, 2, 1, 1], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
    g_b5= tf.get_variable('g_b5', [1], initializer=tf.truncated_normal_initializer(stddev=0.02))
    g5= tf.nn.conv2d(g4, g_w5, strides=[1, 1, 1, 1], padding='SAME')
    g5 = g5 + g_b5
    g5 = tf.nn.leaky_relu(g5)
    
    
    # Dimensions of g4: batch_size x 28 x 28 x 1
    return g5

z_dimensions = 100
batch_size = 50

x_placeholder = tf.placeholder(tf.float32, shape = [None,64,90,1], name='x_placeholder')
# x_placeholder is for feeding input images to the discriminator

GZ= generator(batch_size, z_dimensions)
Gz= tf.matmul(H,tf.transpose(tf.reshape(GZ, [-1, 64*64])))
Gz=tf.reshape(tf.transpose(Gz),[batch_size,90,64,1])
Gz=tf.transpose(Gz,perm=[0,2,1,3])

# GZ holds the generated images,Gz holds measurements

Dx = discriminator(x_placeholder)
# Dx will hold discriminator prediction probabilities
# for the real MNIST images

Dg = discriminator(Gz, reuse_variables=True)
# Dg will hold discriminator prediction probabilities for generated images

#Define losses
d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = Dx, labels = tf.ones_like(Dx)))
d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = Dg, labels = tf.zeros_like(Dg)))
g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = Dg, labels = tf.ones_like(Dg)))


# d_loss = tf.reduce_mean(tf.reduce_sum(tf.square(Dx-1.0)))+tf.reduce_mean(tf.reduce_sum(tf.square(Dg)))

# g_loss =2*tf.reduce_mean(tf.reduce_sum(tf.square(Dg-1.0)))


# d_loss_real = tf.reduce_mean(tf.reduce_sum(tf.square(Dx-1.0)))
# d_loss_fake = tf.reduce_mean(tf.reduce_sum(tf.square(Dg)))

# Define variable lists
tvars = tf.trainable_variables()
d_vars = [var for var in tvars if 'd_' in var.name]
g_vars = [var for var in tvars if 'g_' in var.name]


# Define the optimizers
# Train the discriminator
d_trainer_fake = tf.train.AdamOptimizer(0.0003).minimize(d_loss_fake, var_list=d_vars)
d_trainer_real = tf.train.AdamOptimizer(0.0003).minimize(d_loss_real, var_list=d_vars)

# Train the generator
g_trainer = tf.train.AdamOptimizer(0.0001).minimize(g_loss, var_list=g_vars)

# From this point forward, reuse variables
tf.get_variable_scope().reuse_variables()

sess = tf.Session()

# Send summary statistics to TensorBoard
tf.summary.scalar('Generator_loss', g_loss)
tf.summary.scalar('Discriminator_loss_real', d_loss_real)
tf.summary.scalar('Discriminator_loss_fake', d_loss_fake)

images_for_tensorboard = generator(batch_size, z_dimensions)
tf.summary.image('Generated_images', images_for_tensorboard, 5)
merged = tf.summary.merge_all()
logdir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
writer = tf.summary.FileWriter(logdir, sess.graph)

sess.run(tf.global_variables_initializer())

# Pre-train discriminator
for i in range(3000):
    real_image_batch = input.next_batch(batch_size).reshape([batch_size, 64, 90, 1])
    _, __ = sess.run([d_trainer_real, d_trainer_fake],
                                           {x_placeholder: real_image_batch})
    if i % 1000== 0 or i == 1:
          print('Step %i' % (i))

# Train generator and discriminator together
for i in range(100000):
    real_image_batch = input.next_batch(batch_size).reshape([batch_size, 64, 90, 1])

    # Train discriminator on both real and fake images
    _, __ = sess.run([d_trainer_real, d_trainer_fake],
                                           {x_placeholder: real_image_batch})

    # Train generator
    _ = sess.run(g_trainer)
    
    if i % 1000== 0 or i == 1:
        print('Step %i' % (i))

    if i % 10 == 0:
        # Update TensorBoard with summary statistics
        summary = sess.run(merged, {x_placeholder: real_image_batch})
        writer.add_summary(summary, i)

# Optionally, uncomment the following lines to update the checkpoint files attached to the tutorial.
saver = tf.train.Saver()
saver.save(sess, 'GAN6.6')


