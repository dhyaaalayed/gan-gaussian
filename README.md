# gan-gaussian
An implementation for Vanilla GAN to generate different 5 gaussian distributions
![vanilla_gan_frames](https://user-images.githubusercontent.com/26183913/52636853-01cf1500-2ece-11e9-9144-8e3ee6cd3fee.gif)

Only four distributions have been learned by GAN not all of them. This failure called Mode Collapse.
Unlike WGAN which doesn't collapse when the training data contains many distributions. See [WGAN repository](https://github.com/dhyaaalayed/wgan-gaussian)

# Content:
- GAN Architecture
- Generator Architecture
- Discriminator Architecture
- How to train the model


# Model Overview

GAN consists of a generative model called Generator and a discriminative model called Discriminator
<img width="939" alt="image" src="https://user-images.githubusercontent.com/26183913/52645972-1fa67500-2ee2-11e9-8714-dbab93e21c79.png">

The Generator takes a random vector Z as input to output new data. The Discriminator tries to distinguish between the generated data by the generator and the real data
Our goal is to train the Generator to generate fake data looks like the real data until the discriminator will not be able to distinguish between the real data and the fake data
Both of them try to Faul the other like Minmax game
<img width="593" alt="image" src="https://user-images.githubusercontent.com/26183913/52646669-6f397080-2ee3-11e9-9fa2-118aa61463b3.png">
G tries to maximize the probabilty of the fake data
D tries to minimize the probabilty of the fake data and maximize the probabilty of the real data

# Generator Architecture
It consists of an input layer of 2 neurons for the z vector, 3 hidden layers of 512 neurons and an output layer of 2 neurons
activation functions of the 3 hidden layers are Relus and linear for the output layer 
<img width="453" alt="image" src="https://user-images.githubusercontent.com/26183913/52647282-b4aa6d80-2ee4-11e9-9ac2-7e4aff1ddcce.png">

# Discriminator Architecture
it consists of an input layer of 2 neurons for the training data, 3 hidden layers of 512 neurons of Relu activation function and an output layer of 1 neuron of sigmoid activation function
<img width="460" alt="image" src="https://user-images.githubusercontent.com/26183913/52647390-e9b6c000-2ee4-11e9-804a-a1204f5872c3.png">

# Generator loss function
Generator tries to maximize the probability of the generated data
<img width="237" alt="image" src="https://user-images.githubusercontent.com/26183913/52648830-6ea2d900-2ee7-11e9-9609-b04c49db101d.png">
Tensorflow code of the generator loss function:
```
g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))
```

# Discriminator loss function
Discriminator tries to minimize the probability of the generated data and to maximize the probability of the real data
<img width="421" alt="image" src="https://user-images.githubusercontent.com/26183913/52648447-dc9ad080-2ee6-11e9-9e7e-6200bdfb05a5.png">
Tensorflow code of the discriminator loss function:
```
D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real)))
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))
d_loss = D_loss_real + D_loss_fake
```
# How to use the code:
Write in the console python `vanilla_gan.py` to train the model for generating 5 Gaussian distributions