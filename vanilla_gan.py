import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)



def genGauss(p,n=1,r=1):
    # Load the dataset
    x = []
    y = []
    for k in range(n):
        x_t, y_t = np.random.multivariate_normal([math.sin(2*k*math.pi/n), math.cos(2*k*math.pi/n)], [[0.0125, 0], [0, 0.0125]], p).T
        x.append(x_t)
        y.append(y_t)

    x=np.array(x).flatten()[:,None]
    y=np.array(y).flatten()[:,None]
    x-=np.mean(x)
    y-=np.mean(y)
    train=np.concatenate((x,y),axis=1)

    return train/(np.max(train)*r)



nb_neurons_h1 = 512
nb_neurons_h2 = 512
nb_neurons_h3 = 512
learning_rate =  5e-5
batch_size = 32

z_dim = 2
x_dim = 2
img_dim = 2



z_input_layer = tf.placeholder(tf.float32, shape = [None, z_dim])
x_input_layer = tf.placeholder(tf.float32, shape = [None, x_dim])


# Generator weights and biasis
g_w1 = tf.Variable(xavier_init([z_dim, nb_neurons_h1]), name = 'generator_weights1')
g_b1 = tf.Variable(xavier_init([nb_neurons_h1]), name = 'generator_biases1')
g_w2 = tf.Variable(xavier_init([nb_neurons_h1, nb_neurons_h2]), name = 'generator_weights2')
g_b2 = tf.Variable(xavier_init([nb_neurons_h2]), name = 'generator_biases2')
g_w3 = tf.Variable(xavier_init([nb_neurons_h2, nb_neurons_h3]), name = 'generator_weights3')
g_b3 = tf.Variable(xavier_init([nb_neurons_h3]), name = 'biases')
g_w4 = tf.Variable(xavier_init([nb_neurons_h3, x_dim]), name = 'generator_weights4')
g_b4 = tf.Variable(xavier_init([x_dim]), name = 'generator_biases4')



def Generator(z_input):
    g_y1 = tf.nn.relu((tf.matmul(z_input, g_w1) + g_b1), name = 'generator_activation_layer1')
        
    # Weihgts and biases for the second layer:  
    g_y2 = tf.nn.relu((tf.matmul(g_y1, g_w2) + g_b2) , name = 'generator_activation_layer2')
    
    # Weihgts and biases for the third layer:
    g_y3 = tf.nn.relu((tf.matmul(g_y2, g_w3) + g_b3), name = 'generator_activation_layer3')
    
    # Generator output layer
    tf.matmul(g_y3, g_w4) + g_b4
    return g_y4

theta_g = [g_w1, g_w2, g_w3, g_w4, g_b1, g_b2, g_b3, g_b4]


#### Build the discriminator

# discriminator Variables
d_w1 = tf.Variable(xavier_init([x_dim, nb_neurons_h1]), name = 'discriminator_weights1')
d_b1 = tf.Variable(xavier_init([nb_neurons_h1]), name = 'discriminator_biases1')
d_w2 = tf.Variable(xavier_init([nb_neurons_h1, nb_neurons_h2]), name = 'discriminator_weights2')
d_b2 = tf.Variable(xavier_init([nb_neurons_h2]), name = 'discriminator_biases2')
d_w3 = tf.Variable(xavier_init([nb_neurons_h2, nb_neurons_h3]), name = 'discriminator_weights3')
d_b3 = tf.Variable(xavier_init([nb_neurons_h3]), name = 'discriminator_biases3')
d_w4 = tf.Variable(xavier_init([nb_neurons_h3, 1]), name = 'discriminator_weights4')
d_b4 = tf.Variable(xavier_init([1]), name = 'discriminator_biases4')
theta_d = [d_w1, d_w2, d_w3, d_w4, d_b1, d_b2, d_b3, d_b4]


def Discriminator(x_input):

    # first layer
    d_y1 = tf.nn.relu((tf.matmul(x_input, d_w1) + d_b1), name = 'discriminator_activation_layer1')
    # second layer
    d_y2 = tf.nn.relu((tf.matmul(d_y1, d_w2) + d_b2) , name = 'discriminator_activation_layer2')
    # third layer
    d_y3 = tf.nn.relu((tf.matmul(d_y2, d_w3) + d_b3), name = 'discriminator_activation_layer3')
    # Discriminator output layer
    d_y4 = tf.matmul(d_y3, d_w4) + d_b4
    output = tf.nn.sigmoid(d_y4, name = 'discriminator_output')
    return output, d_y4



fake_img = Generator(z_input_layer)
d_real, D_logit_real = Discriminator(x_input_layer)
d_fake, D_logit_fake = Discriminator(fake_img)


# Disciminator loss function
D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real)))
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))
d_loss = D_loss_real + D_loss_fake

# Generator loss function
g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))



d_optimizer = tf.train.AdamOptimizer(learning_rate).minimize(d_loss, var_list = theta_d)
g_optimizer = tf.train.AdamOptimizer(learning_rate).minimize(g_loss, var_list = theta_g)


sess = tf.Session()
sess.run(tf.initialize_all_variables())

epochs = 10000

X_train = genGauss(100, 5, 0.5)
np.random.shuffle(X_train)
epoch_discriminator_loss = []
epoch_generator_loss = []

fig, axarr = plt.subplots(1, 2, figsize=(12,4 ))

nb_batches = int(X_train.shape[0] / batch_size)
for epoch in range(10000):
    print('Epoch :', epoch)
    batch_discriminator_loss = []
    batch_generator_loss = []
    print('nb_batches: ', nb_batches)
    for i in range(nb_batches):
        image_batch = X_train[i * batch_size:(i + 1) * batch_size]

        _, d_loss_curr = sess.run([d_optimizer, d_loss], feed_dict = {x_input_layer: image_batch, z_input_layer: np.random.uniform(-1, 1, (batch_size, z_dim))})
        _, g_loss_curr = sess.run([g_optimizer, g_loss], feed_dict = {z_input_layer: np.random.uniform(-1, 1, (batch_size, z_dim))})
        batch_discriminator_loss.append(d_loss_curr)
        batch_generator_loss.append(g_loss_curr)
    epoch_generator_loss.append(np.mean(batch_generator_loss))
    epoch_discriminator_loss.append(np.mean(batch_discriminator_loss))
    print('discriminator loss: ', np.mean(batch_generator_loss))
    print('generator loss: ', np.mean(batch_discriminator_loss))

    samples = sess.run(fake_img, feed_dict = {z_input_layer: np.random.uniform(-1, 1, (1000, z_dim))})
    
    fig.suptitle('Vanilla GAN alg: - Epoch: {}'.format(epoch))
    axarr[0].set_title('Real Data vs. Generated Data')
    axarr[0].scatter(X_train[:, 0], X_train[:, 1], c = 'red', label = 'Real data', marker = '.')
    axarr[0].scatter(samples[:, 0], samples[:, 1], c = 'green', label = 'Fake data', marker = '.')
    axarr[0].legend(loc='upper left')
    axarr[1].set_title('Generator & Discriminator error functions')
    axarr[1].plot(epoch_discriminator_loss, color='red', label = 'Discriminator loss')
    axarr[1].plot(epoch_generator_loss, color='blue', label = 'Generator loss')
    axarr[1].legend(loc='upper left')
    fig.savefig('tf_vanilla_gan_results/frame.jpg')
    fig.savefig('tf_vanilla_gan_results/frame' + str(epoch) + '.jpg')
    axarr[0].clear()
    axarr[1].clear()













































