import numpy as np
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.INFO)


class Model(object):

    def __init__(self,training_data,prediction_data,test_pct):    

        self.training_data=training_data
        self.prediction_data=prediction_data       

        self.test_pct=test_pct      

        self.df_pred=self.prediction_data.drop('t_id', axis=1)
        self.df_pred['target']=np.nan
        
        
        self.LABEL_COLUMNS = self.training_data.drop('target', axis=1).columns

    
        test_idx=int(self.test_pct*len(self.training_data))
        
        #test_pct is the percentage of the data used to train, the rest of the data 
        #will be used to evaluate the model
        self.df_train=self.training_data[0:test_idx]
        self.df_test=self.training_data[test_idx+1:len(self.training_data)]
        
        print('Building estimator')
        m = self.build_estimator()
        print('Fitting')
        m.fit(input_fn=self.train_input_fn, steps=500)
        self.results = m.evaluate(input_fn=self.eval_input_fn,steps=100)
                    
        print('Predicting')    
        self.predictions = m.predict(input_fn=self.pred_input_fn)
        
    
    def input_fn(self,df):
        # Creates a dictionary mapping from each continuous feature column name (k) to
        # the values of that column stored in a constant Tensor.
        continuous_cols = {k: tf.constant(df[k].values, shape=[df[k].size, 1]) for k in self.LABEL_COLUMNS}
        
        # Converts the label column into a constant Tensor.
        label = tf.constant(df['target'].values)
        # Returns the feature columns and the label.
        return dict(continuous_cols), label
    
    
    
    def build_estimator(self):
        #simple columns
        contribLayers=[]
        for k in self.LABEL_COLUMNS:
            contribLayers.append(tf.contrib.layers.real_valued_column(k))
            
        #crossed columns
        #crossedLayers=[]
        #for k in LABEL_COLUMNS:
        #    for l in LABEL_COLUMNS:
        #        if k != l:
        #            crossedLayers.append(tf.contrib.layers.crossed_column([k,l], hash_bucket_size=int(1e6)))
        
        wide_columns = contribLayers #+ crossedLayers
        
        print('Building neural network')
        m = tf.contrib.learn.DNNRegressor(feature_columns=wide_columns, hidden_units=[50,1000,100,50], optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.05))
        #,optimizer=tf.train.FtrlOptimizer(learning_rate=0.1,l1_regularization_strength=1.0,l2_regularization_strength=1.0))
        return m
    
    def train_input_fn(self):
        return self.input_fn(self.df_train)
    
    def eval_input_fn(self):
        return self.input_fn(self.df_test)
    
    def pred_input_fn(self):
        return self.input_fn(self.df_pred)
    
    
