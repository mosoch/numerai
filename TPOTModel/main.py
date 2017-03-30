import pandas as pd

import model03 as Model

def main():
    #load data from csv
    print('Parsing CSVs')
    training_data=pd.read_csv('numerai_training_data.csv', header=0)
    prediction_data = pd.read_csv('numerai_tournament_data.csv', header=0)


    #split of data into model building/model testing: 67%/33%
    model_tpot=Model.Model(training_data,prediction_data,0.67)

    #return anticipated model accuracy (hopefully better than 0.5)
    print('Anticipated accuracy is %s' % model_tpot.result)

    #return tournament predictions in .csv form
    print('Writing model predictions to csv')
    write_csv=pd.DataFrame({'t_id':prediction_data['t_id'],'probability':model_tpot.predictions})    
    write_csv.to_csv('numerai_test_02.csv', index=False)

if __name__=="__main__":
    main()