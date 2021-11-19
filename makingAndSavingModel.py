'''
Builds and Trains a Neural Network of 4 layers to recognize handwritten digits
Manages to achieve an average accuracy of 97% with a loss <1%

Author: @asirgue
Version: 2.0 edited on 19/11 before uploading to Github
'''
import tensorflow as tf

# Downloading the data set
mnist = tf.keras.dataset.mnist 

# Splitting labelled data between training data (used to train the model) and testing data (used to assess the model)
(x_train,y_train), (x_text, y_test) = mnist.load_data()

# Normalizingdata (scale it down) because gray scale images have values from 0-255 (if we ignore RGB) and we want each pixel's value to be between 0-1
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.nomalize(x_test, axis=1)

# Model is a basic Sequential model
model = tf.keras.models.Sequential()

# Building a neural network of 4 layers, input layer is a line of 784 pixel gray scale values, then we have two 128 neurons layers, and we end with a 10 neurons layer (1 neuron for each digit from 0 to 9)
model.add(tf.keras.layers.Flatten(input_shape=(28,28))) #Flattening our 28*28 pixels grid to one straight line of 784 pixel gray scale values (from 0 to 1)
model.add(tf.keras.layers.Dense(128,activation = 'relu')) 
model.add(tf.keras.layers.Dense(128,activation = 'relu')) 
model.add(tf.keras.layers.Dense(10,activation = 'softmax')) #Softmax makes sure that the sum of every neuron's value is 1, it's a way to show confidence (value of each of the 10 neuron tells us how likely it is that this value is the one represented by the digit)

#Compiling the model
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

#Training the model
model.train(x_train,y_train, epoch = 3)

model.save('handwritten.model')
