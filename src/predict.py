"""
predict.py

COMPLETAR DOCSTRING

DESCRIPCIÓN:
AUTOR:
FECHA:
"""

# Imports
import pickle
import pandas as pd

class MakePredictionPipeline(object):
    
    def __init__(self, input_path, output_path, model_path: str = None):
        self.input_path = input_path
        self.output_path = output_path
        self.model_path = model_path
                
                
    def load_data(self) -> pd.DataFrame:
        """
        COMPLETAR DOCSTRING
        """
        print(self.input_path)
        data = pd.read_csv(self.input_path + 'test_final.csv',index_col=0)

        return data

    def load_model(self) -> None:
        """
        COMPLETAR DOCSTRING
        """    
        self.model = pickle.load(open(self.model_path+"model.sav",'rb')) # Esta función es genérica, utilizar la función correcta de la biblioteca correspondiente
        
        return None


    def make_predictions(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        COMPLETAR DOCSTRING
        """
        #print(data.head())
        new_data = self.model.predict(data)
    
        return pd.DataFrame(new_data)


    def write_predictions(self, predicted_data: pd.DataFrame) -> None:
        """
        COMPLETAR DOCSTRING
        """
        print(predicted_data)
        predicted_data.to_csv(self.output_path+'data_test.csv')
        return None


    def run(self):

        data = self.load_data()
        self.load_model()
        df_preds = self.make_predictions(data)
        self.write_predictions(df_preds)


if __name__ == "__main__":
    
    #spark = Spark()
    
    pipeline = MakePredictionPipeline(input_path = '../data/output/',
                                      output_path = '../model/output/',
                                      model_path = '../model/')
    pipeline.run()  