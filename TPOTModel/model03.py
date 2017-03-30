from tpot import TPOTClassifier

from sklearn.model_selection import train_test_split
 
class Model(object):
    
    def __init__(self,training_data,prediction_data,test_pct):
        
        self.training_data=training_data
        self.prediction_data=prediction_data   
        self.test_pct=test_pct    
       
        #write training data as one-hot vector
        print('Formatting model training data')
        A = self.training_data['target'].values
        #A=A[0:1000]
        
        B = self.training_data.drop('target', axis=1).values
        #B=B[0:1000]
        
        #model
        print('Testing models')
        X_train, X_test, y_train, y_test = train_test_split(B, A,
                                                            train_size=self.test_pct, test_size=(1-self.test_pct))
        
        tpot = TPOTClassifier(generations=5, population_size=50, scoring='log_loss', verbosity=2)
        tpot.fit(X_train, y_train)
        self.result=tpot.score(X_test, y_test)
        tpot.export('tpot_pipeline_2.py')
                
        C=self.prediction_data.drop('t_id', axis=1).values
        
        self.predictions=tpot.predict(C)
        