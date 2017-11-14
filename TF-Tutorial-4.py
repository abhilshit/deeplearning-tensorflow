
# coding: utf-8

# In[105]:


import os
import tensorflow as tf
import Image
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import numpy as np


# In[106]:


filenames = [];
labels = [];
for filename in os.listdir("/data/tutorial/stop_sign_jpg"):
    if filename.endswith(".jpg"):
        filenames.append("/data/tutorial/stop_sign_jpg/"+filename)
        if filename.startswith("14_"):
            labels.append(1)
        else:
            labels.append(0)


# In[107]:


filenames_test = [];
labels_test = [];
for filename in os.listdir("/data/tutorial/test"):
    if filename.endswith(".jpg"):
        filenames_test.append("/data/tutorial/test/"+filename)
        if filename.startswith("14_"):
            labels_test.append(1)
        else:
            labels_test.append(0)


# In[108]:


image_data = np.array([np.array(Image.open(fname)) for fname in filenames])
print(image_data.shape)
X_train_flatten = image_data.reshape(image_data.shape[0], -1).T
print(X_train_flatten.shape)


# In[109]:


image_data_test = np.array([np.array(Image.open(fname)) for fname in filenames_test])
print(image_data_test.shape)
X_test_flatten = image_data_test.reshape(image_data_test.shape[0], -1).T
print(X_test_flatten.shape)


# In[110]:


#Normalize the vectors
X_train = X_train_flatten / 255
X_test = X_test_flatten / 255


# In[111]:


def one_hot_matrix(labels, C):
    C = tf.constant(C, name='C')
    one_hot_matrix = tf.one_hot(indices=labels, depth=C, axis=0)
    sess = tf.Session()
    one_hot = sess.run(one_hot_matrix)
    sess.close()
    return one_hot


# In[112]:


Y_train = one_hot_matrix(labels, 2)
Y_test = one_hot_matrix(labels_test, 2)


# In[113]:


print(Y_train.shape)
print(Y_test.shape)


# In[128]:


#tf.reset_default_graph()
def create_placeholders(n_x, n_y):
    X = tf.placeholder(tf.float32, [n_x, None], name="X")
    Y = tf.placeholder(tf.float32, [n_y, None], name="Y")
    return X, Y

def initialize_parameters():
    """
    Initializes parameters to build a neural network with tensorflow. The shapes are:
                        W1 : [25, 12288]
                        b1 : [25, 1]
                        W2 : [12, 25]
                        b2 : [12, 1]
                        W3 : [2, 12]
                        b3 : [2, 1]
    
    Returns:
    parameters -- a dictionary of tensors containing W1, b1, W2, b2, W3, b3
    """
    
    #tf.set_random_seed(1)                   
        
    W1 = tf.get_variable("W1", [25, 12288], initializer = tf.contrib.layers.xavier_initializer(seed=1))
    b1 = tf.get_variable("b1", [25, 1], initializer = tf.zeros_initializer())
    W2 = tf.get_variable("W2", [12, 25], initializer = tf.contrib.layers.xavier_initializer(seed=1))
    b2 = tf.get_variable("b2", [12, 1], initializer = tf.zeros_initializer())
    W3 = tf.get_variable("W3", [2, 12], initializer = tf.contrib.layers.xavier_initializer(seed=1))
    b3 = tf.get_variable("b3", [2, 1], initializer = tf.zeros_initializer())
    ### END CODE HERE ###

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}
    
    return parameters


def forward_propagation(X, parameters):
    """
    Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX
    
    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"
                  the shapes are given in initialize_parameters

    Returns:
    Z3 -- the output of the last LINEAR unit
    """
    
    # Retrieve the parameters from the dictionary "parameters" 
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    
    Z1 = tf.add(tf.matmul(W1, X), b1)                      # Z1 = np.dot(W1, X) + b1
    A1 = tf.nn.relu(Z1)                                    # A1 = relu(Z1)
    Z2 = tf.add(tf.matmul(W2, A1), b2)                     # Z2 = np.dot(W2, a1) + b2
    A2 = tf.nn.relu(Z2)                                    # A2 = relu(Z2)
    Z3 = tf.add(tf.matmul(W3, A2), b3)                     # Z3 = np.dot(W3,Z2) + b3
    
    return Z3


def model(X_train, Y_train, X_test, Y_test, learning_rate = 0.0001,
          num_epochs = 2, minibatch_size = 1, print_cost = True):
    """
    Implements a three-layer tensorflow neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SOFTMAX.
    
    """
    graph1 = tf.Graph()
    with graph1.as_default():
       #ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables
        tf.set_random_seed(1)                             # to keep consistent results
        seed = 3                                          # to keep consistent results
        (n_x, m) = X_train.shape                          # (n_x: input size, m : number of examples in the train set)
        n_y = Y_train.shape[0]                            # n_y : output size
        costs = []                                        # To keep track of the cost

        X, Y = create_placeholders(n_x, n_y)
        parameters = initialize_parameters()
        Z3 = forward_propagation(X, parameters)

        # to fit the tensorflow requirement for tf.nn.softmax_cross_entropy_with_logits(...,...)
        logits = tf.transpose(Z3)
        labels = tf.transpose(Y)
        predictions = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        cost = tf.reduce_mean(predictions)

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

        init = tf.global_variables_initializer()

        saver = tf.train.Saver()
        print("Creating Saver")
        
    with tf.Session(graph=graph1) as sess:

        sess.run(init)

        # Do the training loop
        for epoch in range(num_epochs):

            epoch_cost = 0.                       # Defines a cost related to an epoch
            seed = seed + 1
            (minibatch_X, minibatch_Y) = (X_train,Y_train)

            _ , minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})

            epoch_cost += minibatch_cost 

            # Print the cost every epoch
            if print_cost == True and epoch % 100 == 0:
                print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost)

        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        # lets save the parameters in a variable
        parameters = sess.run(parameters)
        print("Parameters have been trained!")

        # Calculate the correct predictions
        correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))

        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        print("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
        print("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))

        saver.save(sess, '/data/tutorial/stop_sign_model2/model')
        return parameters,prediction

parameters = model(X_train, Y_train, X_test, Y_test)

