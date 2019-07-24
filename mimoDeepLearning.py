import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization

k = 4
M = 2**k
N = 7
batch_size = 400
batch_num = 1000
epoch = 20

R = k/N

sigma_train = 0.316

def BN_layer1(input):

    epsilon = 1e-7
    #batch_mean, batch_var = tf.nn.moments(input,[0])
    squar = tf.reduce_mean(tf.square(input) , 1)

    z = (input) / tf.sqrt(tf.reshape(squar, [-1,1]))
    
    
    return z


def BN_layer(input):

    epsilon = 1e-7
    batch_mean, batch_var = tf.nn.moments(input,[0])

    z = (input -batch_mean ) / tf.sqrt(batch_var + epsilon)
    
    
    return z
  

def BLER(inputs, outputs):
    
    e = 0
    for i in range(len(inputs)):
        if outputs[i] != inputs[i]:
            e +=1
    return e/len(inputs)

def prepare_data(batch_size, batch_num,  M , sigma , N):
    
    data1 = np.random.randint(0, M, batch_num*batch_size)
    x_train1 = np_utils.to_categorical(data1)
    
    data2 = np.random.randint(0, M, batch_num*batch_size)
    x_train2 = np_utils.to_categorical(data2)
    
    H11 = pd.DataFrame(np.random.rayleigh(1 ,(batch_size*batch_num, N )))
#    H11 = h11
#    for i in range(N-1):
#        H11 = pd.concat([H11,h11], axis=1)
        
    H12 = pd.DataFrame(np.random.rayleigh(1 ,(batch_size*batch_num, N )))
#    H12 = h12
#    for i in range(N-1):
#        H12 = pd.concat([H12,h12], axis=1)
        
    H21 = pd.DataFrame(np.random.rayleigh(1 ,(batch_size*batch_num, N )))
#    H21 = h21
#    for i in range(N-1):
#        H21 = pd.concat([H21,h21], axis=1)
        
    H22 = pd.DataFrame(np.random.rayleigh(1 ,(batch_size*batch_num, N )))
#    H22 = h22
#    for i in range(N-1):
#        H22 = pd.concat([H22,h22], axis=1)
    
    noise_train1 = np.random.normal(0, sigma , (batch_num*batch_size, N))
    noise_train2= np.random.normal(0, sigma , (batch_num*batch_size, N))
    
    
    return data1, data2, x_train1, x_train2 , noise_train1, noise_train2, np.array(H11), np.array(H12), np.array(H21), np.array(H22)
   
    
data1 ,data2 , x_train1, x_train2, noise_train1, noise_train2, H11_train, H12_train, H21_train, H22_train = prepare_data(batch_size, batch_num, M, sigma_train ,N)


############# Model

x1 =  tf.placeholder(dtype = tf.float32 , shape=(None , M ))
x2 =  tf.placeholder(dtype = tf.float32 , shape=(None , M ))
noise1 = tf.placeholder(dtype = tf.float32, shape=(None ,N))
noise2= tf.placeholder(dtype = tf.float32, shape=(None ,N))

ch11 = tf.placeholder(dtype = tf.float32, shape = (None, N))
ch12 = tf.placeholder(dtype = tf.float32, shape = (None, N))
ch21 = tf.placeholder(dtype = tf.float32, shape = (None, N))
ch22 = tf.placeholder(dtype = tf.float32, shape = (None, N))

h11 = tf.reshape(ch11[:,0],[-1,1])
h12 = tf.reshape(ch12[:,0],[-1,1])
h21 = tf.reshape(ch21[:,0],[-1,1])
h22 = tf.reshape(ch22[:,0],[-1,1])

concat_layer = tf.concat([x1,x2,ch11,ch12,ch21,ch22],1)
#concat_layer = tf.concat([x1,x2],1)


#########################transmiters

#### first Dense layer
w1 = tf.Variable(tf.random_uniform(shape=(2*M+4*N , 3*M) , minval=-0.1 , maxval=0.1))
b1 = tf.Variable( tf.zeros(3*M))
layer1 = tf.matmul(concat_layer, w1)+b1
relu1 = tf.nn.relu(layer1)

##### second Dense layer
#w2 = tf.Variable( tf.random_uniform(shape=(2*M, M), minval=-0.1, maxval=-0.1 ))
#b2 = tf.Variable(tf.zeros(M))
#layer2 = tf.matmul(relu1 , w2) + b2
#relu2 = tf.nn.relu(layer2)
#

##### 3th Dense layer
w3 = tf.Variable( tf.random_uniform(shape=(3*M, 2*M), minval=-0.1, maxval=-0.1 ))
b3 = tf.Variable(tf.zeros(2*M))
layer3 = tf.matmul(relu1 , w3) + b3
relu3 = tf.nn.relu(layer3)


#########################transmiter1
w3_t1 = tf.Variable( tf.random_uniform(shape=(3*M, N), minval=-0.1, maxval=-0.1 ))
b3_t1 = tf.Variable(tf.zeros(N))
layer3_t1 = tf.matmul(relu1 , w3_t1) + b3_t1

#BN_t1 = tf.layers.batch_normalization(layer3_t1)
BN_t1 = BN_layer1(layer3_t1)
#BN_t1 = BatchNormalization()(layer3_t1)


#########################transmiter2
w3_t2 = tf.Variable( tf.random_uniform(shape=(3*M, N), minval=-0.1, maxval=-0.1 ))
b3_t2 = tf.Variable(tf.zeros(N))
layer3_t2 = tf.matmul(relu1 , w3_t2) + b3_t2

#BN_t2 = tf.layers.batch_normalization(layer3_t2)
BN_t2 = BN_layer1(layer3_t2)
#BN_t2 = BatchNormalization()(layer3_t2)

####################### reciver1

faded_layer1 = BN_t1 * ch11 + BN_t2 * ch21
#print(faded_layer1.shape)
#### Noise layer
Noise_layer1 = faded_layer1 + noise1


w1_3 = tf.Variable( tf.random_uniform(shape=(N , M), minval=-0.1, maxval=0.1 ) )
b1_3 = tf.Variable(tf.zeros(M))
layer1_3 = tf.matmul(Noise_layer1 , w1_3)+b1_3
relu1_3 = tf.nn.relu(layer1_3)


#######
w1_4 = tf.Variable( tf.random_uniform(shape=(M , M) , minval=-0.1, maxval=0.1))
b1_4 = tf.Variable(tf.zeros(M))
layer1_4 = tf.matmul(relu1_3, w1_4)+b1_4

###### softmax layer
output1 = tf.nn.softmax(layer1_4)

#### loss
loss1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=x1,logits=layer1_4))

####################### reciver2

faded_layer2 = BN_t2 * ch22 + BN_t1 * ch12
#### Noise layer
Noise_layer2 = faded_layer2 + noise2


w2_3 = tf.Variable( tf.random_uniform(shape=(N , M), minval=-0.1, maxval=0.1 ) )
b2_3 = tf.Variable(tf.zeros(M))
layer2_3 = tf.matmul(Noise_layer2 , w2_3)+b2_3
relu2_3 = tf.nn.relu(layer2_3)

#######
w2_4 = tf.Variable( tf.random_uniform(shape=(M , M) , minval=-0.1, maxval=0.1))
b2_4 = tf.Variable(tf.zeros(M))
layer2_4 = tf.matmul(relu2_3, w2_4)+b2_4

###### softmax layer
output2 = tf.nn.softmax(layer2_4)

#### loss
loss2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=x2,logits=layer2_4))



total_loss = (loss1 + loss2)/2

####optimizer
train=tf.train.AdamOptimizer(0.005).minimize(total_loss)


#### accuracy
output_data1 = tf.arg_max(output1 ,1)
output_data2 = tf.arg_max(output2 ,1)



sess = tf.Session()
sess.run(tf.global_variables_initializer())



#### train

for i in range(epoch):
    for j in range(batch_num):
        
        inputs1 = x_train1[j*batch_size:(j+1)*batch_size]
        inputs2 = x_train2[j*batch_size:(j+1)*batch_size]
        Noise1 = noise_train1[j*batch_size:(j+1)*batch_size]
        Noise2 = noise_train2[j*batch_size:(j+1)*batch_size]
        channel11 = H11_train[j*batch_size:(j+1)*batch_size]
        channel12 = H12_train[j*batch_size:(j+1)*batch_size]
        channel21 = H21_train[j*batch_size:(j+1)*batch_size]
        channel22 = H22_train[j*batch_size:(j+1)*batch_size]
        
        sess.run(train, feed_dict={x1:inputs1, x2:inputs2, noise1:Noise1, noise2:Noise2, ch11:channel11, ch12:channel12, ch21:channel21, ch22:channel22})
        
    l = sess.run(total_loss, feed_dict={x1:inputs1, x2:inputs2, noise1:Noise1, noise2:Noise2, ch11:channel11, ch12:channel12, ch21:channel21, ch22:channel22})
    print(i, l)

















######## plot BLER per SNR
SNR_db = np.linspace(-4,50,16)
SNR = 10**(SNR_db/10)
SIGMA = np.sqrt(1/(SNR))    
E1 = []
E2 = []

for i in range(len(SNR)):
    data_test1, data_test2 , x_test1, x_test2, noise_test1, noise_test2, H11_test, H12_test, H21_test, H22_test = prepare_data(1000 , 200 ,M, SIGMA[i] ,N)
    
    out1 = sess.run(output_data1, feed_dict={x1:x_test1, x2:x_test2, noise1:noise_test1, noise2:noise_test2, ch11:H11_test, ch12:H12_test, ch21:H21_test, ch22:H22_test})
    bler = BLER(data_test1 ,out1 )
    E1.append(bler)
    
    out2 = sess.run(output_data2, feed_dict={x1:x_test1, x2:x_test2, noise1:noise_test1, noise2:noise_test2, ch11:H11_test, ch12:H12_test, ch21:H21_test, ch22:H22_test})
    bler2 = BLER(data_test2 ,out2 )
    
    E2.append(bler2)
    
E = (np.array(E1)+np.array(E2))/2.
plt.semilogy(SNR_db , E)


#plt.semilogy(SNR_db , E1)
#plt.semilogy(SNR_db , E2)
plt.grid(True)








sigma_test = 1
data_test1, data_test2 , x_test1, x_test2, noise_test1, noise_test2, H11_test, H12_test, H21_test, H22_test = prepare_data(100 , 10 ,M, SIGMA[i] ,N)

concat_layer
a_concat_layer = sess.run(concat_layer, feed_dict={x1:x_test1, x2:x_test2, noise1:noise_test1, noise2:noise_test2, ch11:H11_test, ch12:H12_test, ch21:H21_test, ch22:H22_test})

a_layer1 = sess.run(layer1, feed_dict={x1:x_test1, x2:x_test2, noise1:noise_test1, noise2:noise_test2, ch11:H11_test, ch12:H12_test, ch21:H21_test, ch22:H22_test})
a_relu1 = sess.run(relu1, feed_dict={x1:x_test1, x2:x_test2, noise1:noise_test1, noise2:noise_test2, ch11:H11_test, ch12:H12_test, ch21:H21_test, ch22:H22_test})
a_layer3 = sess.run(layer3, feed_dict={x1:x_test1, x2:x_test2, noise1:noise_test1, noise2:noise_test2, ch11:H11_test, ch12:H12_test, ch21:H21_test, ch22:H22_test})
a_BN_layer_t1 = sess.run(BN_t1, feed_dict={x1:x_test1, x2:x_test2, noise1:noise_test1, noise2:noise_test2, ch11:H11_test, ch12:H12_test, ch21:H21_test, ch22:H22_test})
a_BN_layer_t2 = sess.run(BN_t2, feed_dict={x1:x_test1, x2:x_test2, noise1:noise_test1, noise2:noise_test2, ch11:H11_test, ch12:H12_test, ch21:H21_test, ch22:H22_test})

a_faded_layer1 = sess.run(faded_layer1, feed_dict={x1:x_test1, x2:x_test2, noise1:noise_test1, noise2:noise_test2, ch11:H11_test, ch12:H12_test, ch21:H21_test, ch22:H22_test})
a_faded_layer2 = sess.run(faded_layer2, feed_dict={x1:x_test1, x2:x_test2, noise1:noise_test1, noise2:noise_test2, ch11:H11_test, ch12:H12_test, ch21:H21_test, ch22:H22_test})


a_layer1_3 = sess.run(layer1_3, feed_dict={x1:x_test1, x2:x_test2, noise1:noise_test1, noise2:noise_test2, ch11:H11_test, ch12:H12_test, ch21:H21_test, ch22:H22_test})
a_relu1_3 = sess.run(relu1_3, feed_dict={x1:x_test1, x2:x_test2, noise1:noise_test1, noise2:noise_test2, ch11:H11_test, ch12:H12_test, ch21:H21_test, ch22:H22_test})
a_layer1_4 = sess.run(layer1_4, feed_dict={x1:x_test1, x2:x_test2, noise1:noise_test1, noise2:noise_test2, ch11:H11_test, ch12:H12_test, ch21:H21_test, ch22:H22_test})

a_layer2_3 = sess.run(layer2_3, feed_dict={x1:x_test1, x2:x_test2, noise1:noise_test1, noise2:noise_test2, ch11:H11_test, ch12:H12_test, ch21:H21_test, ch22:H22_test})
a_relu2_3 = sess.run(relu2_3, feed_dict={x1:x_test1, x2:x_test2, noise1:noise_test1, noise2:noise_test2, ch11:H11_test, ch12:H12_test, ch21:H21_test, ch22:H22_test})
a_layer2_4 = sess.run(layer2_4, feed_dict={x1:x_test1, x2:x_test2, noise1:noise_test1, noise2:noise_test2, ch11:H11_test, ch12:H12_test, ch21:H21_test, ch22:H22_test})

a_output1 = sess.run(output1, feed_dict={x1:x_test1, x2:x_test2, noise1:noise_test1, noise2:noise_test2, ch11:H11_test, ch12:H12_test, ch21:H21_test, ch22:H22_test})
a_output2 = sess.run(output2, feed_dict={x1:x_test1, x2:x_test2, noise1:noise_test1, noise2:noise_test2, ch11:H11_test, ch12:H12_test, ch21:H21_test, ch22:H22_test})

a_output_data1 = sess.run(output_data1, feed_dict={x1:x_test1, x2:x_test2, noise1:noise_test1, noise2:noise_test2, ch11:H11_test, ch12:H12_test, ch21:H21_test, ch22:H22_test})
a_output_data2 = sess.run(output_data2, feed_dict={x1:x_test1, x2:x_test2, noise1:noise_test1, noise2:noise_test2, ch11:H11_test, ch12:H12_test, ch21:H21_test, ch22:H22_test})


a_Noise_layer2 = sess.run(Noise_layer2, feed_dict={x1:x_test1, x2:x_test2, noise1:noise_test1, noise2:noise_test2, ch11:H11_test, ch12:H12_test, ch21:H21_test, ch22:H22_test})
a_layer2_3 = sess.run(layer2_3, feed_dict={x1:x_test1, x2:x_test2, noise1:noise_test1, noise2:noise_test2, ch11:H11_test, ch12:H12_test, ch21:H21_test, ch22:H22_test})
a_relu2_3 = sess.run(relu2_3, feed_dict={x1:x_test1, x2:x_test2, noise1:noise_test1, noise2:noise_test2, ch11:H11_test, ch12:H12_test, ch21:H21_test, ch22:H22_test})
a_layer2_4 = sess.run(layer2_4, feed_dict={x1:x_test1, x2:x_test2, noise1:noise_test1, noise2:noise_test2, ch11:H11_test, ch12:H12_test, ch21:H21_test, ch22:H22_test})
a_output2 = sess.run(output2, feed_dict={x1:x_test1, x2:x_test2, noise1:noise_test1, noise2:noise_test2, ch11:H11_test, ch12:H12_test, ch21:H21_test, ch22:H22_test})
a_output_data2 = sess.run(output_data2, feed_dict={x1:x_test1, x2:x_test2, noise1:noise_test1, noise2:noise_test2, ch11:H11_test, ch12:H12_test, ch21:H21_test, ch22:H22_test})



bler1 = BLER(data_test1 ,a_output_data1 )
bler2 = BLER(data_test2 ,a_output_data2 )

squar = np.square(a_BN_layer_t1)
squar_mean = np.mean(squar,1)
p = np.mean(squar_mean,0)






print('AE_MIMO_CSI')




