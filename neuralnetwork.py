import numpy as np
from numpy.core.fromnumeric import argmax, shape 



class NeuralNetwork:
    def __init__(self, layer_sizes):
        weight_shapes = [(a, b) for a, b in zip(layer_sizes[1:], layer_sizes[:-1])]
        self.weights = [np.random.standard_normal(s) / np.sqrt(s[1]) for s in weight_shapes]
        self.biases = [np.zeros((s, 1)) for s in layer_sizes[1:]]
        self.num_layers = len(layer_sizes)
        self.activations = np.asarray([np.zeros(size,dtype=np.float64) for size in layer_sizes],dtype=object)
    def set_test(self,data_in,labels_in):
        self.test_images=data_in
        self.test_labels=labels_in
        
    def feedforward(self,sample):
        #takes in the input as the activations for the first layer of neurons
        self.activations[0]=sample
        #goes through the remaining layers of the network
        for i in range (0,self.num_layers-1):
            #multiplies matricies in the form z=w_i*x_i+b_i
            z = np.matmul(self.weights[i],self.activations[i])+self.biases[i]
            #apply the activation function to the linear one so a=f(z)
            self.activations[i+1] = self.activation(z)
        #returns the activations of the final layer
        return self.activations[-1]

    def train_sgd(self,training_images, training_labels, epochs, batch_size, learning_rate):
        print('epochs:{0}, batch_size{1}, learning_rate{2}, shape{3}'.format(epochs,batch_size,learning_rate,self.activations.shape))
        training_data = [(x,y) for x,y in zip(training_images, training_labels)]
        print("Start Training")

        for i in range(epochs):
            #creates an array of batches of training data of size "batch_size" 
            batches = [training_data[k:k+batch_size] for k in range(0,len(training_data),batch_size) ]

            for batch in batches:
                self.update_batch(batch,learning_rate)
            #prints after the ith epoch is complete
            print("epoch {0}: {1}".format(i,self.accuracy(self.test_images,self.test_labels)))
            print("epoch {0} complete".format(i))    

    def update_batch(self,batch,learning_rate):
        '''Calculate direction of gradients for a batch and apply the changes to the weights and biases'''
        bias_gradients = [np.zeros(n.shape) for n in self.biases]
        weight_gradients=[np.zeros(n.shape) for n in self.weights]

        #Back propogates for multiple samples then uses the calculated gradients by summing them
        for sample,label in batch:
            bias_deltas,weight_deltas=self.back_propagation(sample, label)


            bias_gradients =[b_gradient +b_delta for b_gradient,b_delta in zip(bias_gradients,bias_deltas)]
            weight_gradients =[w_gradient +w_delta for w_gradient,w_delta in zip(weight_gradients,weight_deltas)]
        len_batch=len(batch)
        #update weights and biases by taking the average adjustment for bias
        #uses the learning rate which is the magniture of shifts of weights and biases
        # (b_gradient/len_batch) is the average direction for that particular neurons biases for that batch
        #we then subtract the original value learning rate * the avg gradient
        self.biases=[b-(learning_rate)*(b_gradient/len_batch) for b,b_gradient in zip(self.biases,bias_gradients)]
        #repeat the same for weights
        self.weights=[w - (learning_rate / len(batch)) * w_gradient for w, w_gradient in zip(self.weights, weight_gradients)]
    
    ''' def back_propagation(self,sample,label):
        #temporary values for bias deltas
        bias_deltas=[np.zeros(b.shape) for b in self.biases]
        weight_deltas=[np.zeros(w.shape) for w in self.biases]
        #computes activations for network
        self.feedforward(sample)
        #set index of the layer l to -1 to start from the back of network
        l=-1
        #∂Δ∇
        #C = cost function
        #a = activation of neuron a = f(z)
        #w = weight 
        #b = bias
        #z = linear input into the activation function (z=wx+b)
        #x = input actiavtion from previous layer (x=a_n-1)
        #(∂C/∂a)*(∂a/∂z) is named delta 2 of 3 parts of the chain rule


        delta = self.cost_derivative(self.activations[l],label)*self.activation_derivative(self.activations[l])

        #∂C/∂b = (∂C/∂a)*(∂a/∂z)*(∂z/∂b) and ∂z/∂b = 1 beacuse 1 is the derivative of z=wx+b wrt b
        bias_deltas[l]=delta*1
        #∂C/∂w = (∂C/∂a)*(∂a/∂z)*(∂z/∂w) and ∂z/∂w = x
        weight_deltas[l]=np.dot(delta,self.activations[l-1].T) # you need to transpose to calculate the dot product
        for l in range (l-1,-self.num_layers+1,-1):
            delta=np()
        return bias_deltas, weight_deltas'''
    def back_propagation(self, sample, label):
        """
        Feed a sample through the network and calculate the changes in weights and biases
        by propagating back through the network in such a way that the cost is minimized
        """
        bias_deltas = [np.zeros(b.shape) for b in self.biases]
        weight_deltas = [np.zeros(w.shape) for w in self.weights]

        # calculate the activations for all neurons by feeding a sample through the network
        self.feedforward(sample)

        # theory by 3Blue1Brown: https://youtu.be/tIeHLnjs5U8
        L = -1

        # 'partial_deltas' is two (of three) parts of the 'chain rule'
        # - the change in cost given a change in a(L) (= cost_function'(a(L), y))
        # - the change in a(L) given a change in z(L) (= sigmoid'(z(L)))
        partial_deltas = self.cost_derivative(self.activations[L], label) * \
                         self.activation_derivative(self.activations[L])

        # now multiply with the third part of the chain rule,
        # this third part is different for weights and biases
        # for biases: the change in z(L) given a change in b(L) = 1         [5:46 in the video]
        # for weights: the change in z(L) given a change in w(L) = a(L-1)   [5:11 in the video]
        bias_deltas[L] = partial_deltas  # * 1 (but this is redundant)
        weight_deltas[L] = np.dot(partial_deltas, self.activations[L - 1].T)

        # continue back propagation, stop at second to last layer, we don't want to adjust the input layer :-)
        while L > -self.num_layers + 1:
            # to update the partial_deltas for a previous activation layer we need to multiply again with a third part
            # for the previous activation: the change in z(L) given a change in a(L-1) = w(L)   [6:05 in the video]
            previous_layer_deltas = np.dot(self.weights[L].T, partial_deltas)

            # again calculate the two (of three) parts of the 'chain rule' for the biases and weights
            # and apply the third part separate
            partial_deltas = previous_layer_deltas * self.activation_derivative(self.activations[L - 1])
            bias_deltas[L - 1] = partial_deltas  # * 1 (but this is redundant)
            weight_deltas[L - 1] = np.dot(partial_deltas, self.activations[L - 2].T)

            L -= 1
            # this loop can be slightly optimized, but it's done this way to stay in line with the theory

        return bias_deltas, weight_deltas

    def accuracy(self, images,labels):
        predictions = self.feedforward(images)
        num_correct = sum(np.argmax(a) == np.argmax(b) for a,b in zip(predictions,labels))
        return ("{0}/{1} accuracy:{2}%".format(num_correct,len(images),num_correct*100/len(images)))
    
    def mean_absolute_error(self, images,labels):
        predictions = self.feedforward(images)
        print("\n"+"Mean Absolute Error :", self.loss(labels,predictions)/len(labels))


    #activation function and derivative ,can be swapped out
    @staticmethod
    def activation(x):
        return 1/(1+np.exp(-x))
    @staticmethod
    def activation_derivative(x):
        return sigmoid_prime(x)
    @staticmethod
    def cost(x,y):
        return sum_of_squares(x,y) 
    #cost functon and derivative can be swapped outxsara
    @staticmethod
    def cost_derivative(x,y):
        return sum_of_squares_prime(x,y)
        



def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_prime(x):
    """
    the derivative of the sigmoid function
    is actually 'sigmoid(x) * (1 - sigmoid(x))'
    but if we take sigmoid(x) as input,
    we can just use the following:
    """
    return sigmoid(x) * (1 - sigmoid(x))
def sum_of_squares(output, y):
    return sum((a - b) ** 2 for a, b in zip(output, y))[0]
def sum_of_squares_prime(output, y):
    return 2 * (output - y)