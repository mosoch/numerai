import pandas as pd
import numpy as np

import model04 as Model

def main():
    #load data from csv
    print('Parsing CSVs')
    training_data=pd.read_csv('numerai_training_data.csv', header=0)[0:100]
    prediction_data = pd.read_csv('numerai_tournament_data.csv', header=0)


    #split of data into model building/model testing: 67%/33%
    model_contrib_learn=Model.Model(training_data,prediction_data,0.67)

    #return anticipated model accuracy (hopefully better than 0.5)
    for key in sorted(model_contrib_learn.results):
        print("%s: %s" % (key, model_contrib_learn.results[key]))

    #return tournament predictions in .csv form
    print('Writing model predictions to csv')
    write_csv=pd.DataFrame({'t_id':prediction_data['t_id'],'probability':np.array(list(model_contrib_learn.predictions))})    
    write_csv.to_csv('numerai_test_03.csv', index=False)

if __name__=="__main__":
    main()