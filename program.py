import neuralnetwork as nn
import numpy as np
import matplotlib.pyplot as plt


#loading the mnist data sets training images
with np.load('mnist.npz') as data: 
	training_images = data['training_images']

#loading the mnist data sets training labels
with np.load('mnist.npz') as data:    
    training_labels = data['training_labels']


#There are 784 pixels in each image because each image is 28x28 hence 784 input neurons
#there are then 10 output neurons corresponding to digits 0 to 9
layer_sizes=(784,100,10)
training_set_size=40000

train_images=training_images[:training_set_size]
train_labels=training_labels[:training_set_size]

test_images=training_images[training_set_size:]
test_labels=training_labels[training_set_size:]

#Initialise a neural network with layer sizes corresponding to the values in layer_sizes
net=nn.NeuralNetwork(layer_sizes)


net.set_test(test_images,test_labels)
#generates outputs

net.train_sgd(train_images,train_labels,epochs=30,batch_size=10,learning_rate=0.25)
#outputs the index of the neuron on the final layer with the highest activation(corresponds to the number)
#print("The neural network outputs",np.argmax(prediction[0]))
#outputs the label of the image from the mnist data set
#print("The correct label is",np.argmax(training_labels[0]))

#evaluate performance without training
#net.print_accuracy(test_images,test_labels)

#net.mean_absolute_error(test_images,test_labels)
