import pandas as pd
import numpy as np
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.INFO)

test_pct=0.67

print('Reading CSV')
training_data = pd.read_csv('numerai_training_data.csv', header=0)
prediction_data = pd.read_csv('numerai_tournament_data.csv', header=0)

df_pred=prediction_data.drop('t_id', axis=1)
df_pred['target']=np.nan


LABEL_COLUMNS = training_data.drop('target', axis=1).columns

def input_fn(df):
    # Creates a dictionary mapping from each continuous feature column name (k) to
    # the values of that column stored in a constant Tensor.
    continuous_cols = {k: tf.constant(df[k].values, shape=[df[k].size, 1]) for k in LABEL_COLUMNS}
    
    # Converts the label column into a constant Tensor.
    label = tf.constant(df['target'].values)
    # Returns the feature columns and the label.
    return dict(continuous_cols), label



def build_estimator():
    #simple columns
    contribLayers=[]
    for k in LABEL_COLUMNS:
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

def train_input_fn():
    return input_fn(df_train)

def eval_input_fn():
    return input_fn(df_test)

def pred_input_fn():
    return input_fn(df_pred)

test_pct=0.67

test_idx=int(test_pct*len(training_data))

#test_pct is the percentage of the data used to train, the rest of the data 
#will be used to evaluate the model
df_train=training_data[0:test_idx]
df_test=training_data[test_idx+1:len(training_data)]

print('Building estimator')
m = build_estimator()
print('Fitting')
m.fit(input_fn=train_input_fn, steps=500)
results = m.evaluate(input_fn=eval_input_fn,steps=100)

for key in sorted(results):
    print("%s: %s" % (key, results[key]))
    
print('Predicting')    
predictions = m.predict(input_fn=pred_input_fn)

write_csv=pd.DataFrame({'t_id':prediction_data['t_id'],'probability':np.array(list(predictions))})
write_csv.to_csv('numerai_test_06.csv', index=False)

