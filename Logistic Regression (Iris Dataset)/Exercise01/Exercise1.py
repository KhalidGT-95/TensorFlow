from __future__ import print_function

from sklearn.cross_validation import train_test_split
import tensorflow as tf
import sklearn
from sklearn import datasets
import numpy as np


iris = datasets.load_iris()

data = iris.data

# Create a numpy array to store the classes of the data
label = np.zeros((150, 3), dtype=np.int)

# Directory to save the tensorboard file
LOGDIR = "/home/khalid/tensorflow/khalid1/"

for i in range(150):
    label[i][iris.target[i]] = 1

labels = label


# Split the data into training and testing samples
data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=0.10, random_state=42)

# Parameters
learning_rate = 0.02
epochs = 300   
display_step = 1

increment = 9

# tf Graph Input
Parameters = tf.placeholder(tf.float32, [None,4]) # mnist data image of shape 28*28=784
Classes = tf.placeholder(tf.float32, [None,3]) # 0-9 digits recognition => 10 classes

# Set model weights
Weight = tf.Variable(tf.random_normal([4,3],mean=0.0,stddev=0.05))
Bias = tf.Variable(tf.zeros([3]))

# Construct model
prediction = tf.nn.softmax(tf.matmul(Parameters, Weight) + Bias) # Softmax

# Minimize error using cross entropy
cost = tf.reduce_mean(-tf.reduce_sum(Classes*tf.log(prediction), reduction_indices=1))

# Gradient Descent
GD_optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Test model
correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(Classes, 1))
# Calculate accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
# Initializing the variables
init = tf.global_variables_initializer()

tf.summary.scalar("loss", cost)

'''
# Make a normal distribution, with a shifting mean
mean_moving_normal = tf.random_normal(shape=[1000], mean=(5*k), stddev=1)
'''

'''
# Record that distribution into a histogram summary
summaries = tf.summary.histogram("normal/moving_mean", mean_moving_normal)
'''

tf.summary.histogram('accuracy',accuracy)
tf.summary.histogram("loss", cost)

# Create a summary to monitor accuracy tensor
tf.summary.scalar("accuracy", accuracy)

# Merge all summaries into a single op
merged_summary_op = tf.summary.merge_all()  
  
# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    
    summary_writer = tf.summary.FileWriter(LOGDIR+'2', graph=tf.get_default_graph())
    
    total_batch = 15    
    
    '''
    # Setup a loop and write the summaries to disk
    N = 400
    for step in range(N):
        k_val = step/float(N)
        summ = sess.run(summaries, feed_dict={k: k_val})
        writer.add_summary(summ, global_step=step)
    '''

    # Training cycle
    for epoch in range(epochs):
        avg_cost = 0.
        
        batch_size = 0      
        
        # Loop over all batches
        for i in range(total_batch):
            #print("mmmm")
            #batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            Parameter_batch = data_train[batch_size:batch_size+increment]
            Classes_batch = labels_train[batch_size:batch_size+increment]
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c, summary = sess.run([GD_optimizer, cost, merged_summary_op], feed_dict={Parameters: Parameter_batch,Classes: Classes_batch})
            # Compute average loss
          
            avg_cost += c / total_batch
            
            
            summary_writer.add_summary(summary, epoch * total_batch + i)
            
            
            '''if i % 10 == 0:
                training = {x:batch_xs,y:batch_ys}
                loss_step = cost.eval(training)
                train_accuracy = accuracy.eval(training)
                print('  step, loss, accurary = %6d: %8.3f,%8.3f' % (i, loss_step, train_accuracy))
            '''
            batch_size += increment
        # Display logs per epoch step
        if (epoch+1) % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

    print("\n\nOptimization Finished!")
    
    print("\n\nNow Testing on Test Set")
    
    print("\n\nAccuracy on Test Set is : ", accuracy.eval({Parameters: data_test, Classes: labels_test}))
    
    print("\n\n")
