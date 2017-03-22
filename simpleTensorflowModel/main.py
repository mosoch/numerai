import pandas as pd

import model02 as Model

def main():
    #load data from csv
    print('Parsing CSVs')
    training_data=pd.read_csv('numerai_training_data.csv', header=0)
    prediction_data = pd.read_csv('numerai_tournament_data.csv', header=0)

    #size of the random batch we are using to build the model: 100
    #number of random batches we'll try: 1000
    #gradient descent: 0.5
    #split of data into model building/model testing: 67%/33%
    model_simple_neural_network=Model.Model(100,1000,0.5,0.67,training_data,prediction_data)

    #return anticipated model accuracy (hopefully better than 0.5)
    print('Anticipated accuracy is %s' % model_simple_neural_network.result)

    #return tournament predictions in .csv form
    print('Writing model predictions to csv')
    write_csv=pd.DataFrame({'t_id':prediction_data['t_id'],'probability':model_simple_neural_network.predictions})    
    write_csv.to_csv('numerai_test_02.csv', index=False)

if __name__=="__main__":
    main()
