import numpy as np

import tensorflow as tf


class Model(object):

    def __init__(self,batch_len,batch_nums,grad_descent,test_pct,training_data,prediction_data):    

        self.training_data=training_data
        self.prediction_data=prediction_data       
        self.batch_len=batch_len
        self.batch_nums=batch_nums
        self.grad_descent=grad_descent
        self.test_pct=test_pct        
        
        #write training data as one-hot vector
        print('Formatting model training data')
        A_init = self.training_data['target'].values
        A=np.zeros([len(A_init),2])
        for i in range(len(A_init)):
            if A_init[i]==0:
                A[i][0]=1
            else:
                A[i][1]=1
        
        B = self.training_data.drop('target', axis=1).values
     
    
        #format tournament data
        print('Formatting tournament data')
       
        C = self.prediction_data.drop('t_id', axis=1).values    
    
   

        #set up model
        print('Setting up model')
        test_idx=int(self.test_pct*len(A))
        
        #test_pct is the percentage of the data used to train, the rest of the data 
        #will be used to evaluate the model
        A_train=A[0:test_idx]
        A_test=A[test_idx+1:len(A)]
        B_train=B[0:test_idx]
        B_test=B[test_idx+1:len(A)]

        

        x = tf.placeholder(tf.float32, [None, 50])
        
        W = tf.Variable(tf.zeros([50, 2]))
        b = tf.Variable(tf.zeros([2]))
        
        y = tf.nn.softmax(tf.matmul(x, W) + b)
    
        y_ = tf.placeholder(tf.float32, [None, 2])
     
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
               
        train_step = tf.train.GradientDescentOptimizer(self.grad_descent).minimize(cross_entropy)
        
        #open tensorflow session
        print('Opening tensorflow session')
        sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()
    
        #get random batch
        for i in range(self.batch_nums):
            #if i%batch_len==0:
            #    print('Finished %d cases' % i)
            batch_i=np.random.randint(0,len(A_train),self.batch_len)     
            batch_xs=np.array([B_train[j] for j in batch_i])
            batch_ys=np.array([A_train[j] for j in batch_i])
            #run session with 100 randomly selected cases       
            sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
        
        correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
      
        self.result = sess.run(accuracy, feed_dict={x: B_test, y_: A_test})
       
        #predictions for tournament data

        self.predictions = sess.run(y[:,1], feed_dict={x: C})
    