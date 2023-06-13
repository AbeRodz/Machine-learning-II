"""
train.py

COMPLETAR DOCSTRING

DESCRIPCIÓN:
AUTOR:
FECHA:
"""

# Imports
import pandas as pd
from sklearn.model_selection import train_test_split, cross_validate, cross_val_score
from sklearn import metrics
from sklearn.linear_model import LinearRegression
import logging

from feature_engineering import FeatureEngineeringPipeline

logging.basicConfig(
    level=logging.INFO, 
     format= '[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s',
     datefmt='%H:%M:%S')

class ModelTrainingPipeline(object):

    def __init__(self, input_path, model_path):
        
        self.input_path = input_path
        self.model_path = model_path

    def read_data(self) -> pd.DataFrame:
        import os
        """
        COMPLETAR DOCSTRING 
        
        :return pandas_df: The desired DataLake table as a DataFrame
        :rtype: pd.DataFrame
        """
            
        # COMPLETAR CON CÓDIGO
        items  = os.listdir(self.input_path)

        for dataset in items:
            if dataset.lower().startswith('test') :
                test_path = dataset
            if  dataset.lower().startswith('train'):
                train_path = dataset

        data_train = pd.read_csv( self.input_path + train_path,index_col=0)
        #data_test = pd.read_csv( self.input_path +test_path)
        # Identificando la data de train y de test, para posteriormente unión y separación
        #pandas_df = pd.concat([data_train], ignore_index=True, sort=False)
        return data_train

    
    def model_training(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        COMPLETAR DOCSTRING
        
        """
        
        # COMPLETAR CON CÓDIGO
        seed = 28
        model = LinearRegression()
        print(df.isna().sum())
        # División de dataset de entrenaimento y validación
        X = df.drop(columns='Item_Outlet_Sales') #[['Item_Weight', 'Item_MRP', 'Outlet_Establishment_Year', 'Outlet_Size', 'Outlet_Location_Type']] # .drop(columns='Item_Outlet_Sales')
        x_train, x_val, y_train, y_val = train_test_split(X, df['Item_Outlet_Sales'], test_size = 0.3, random_state=seed)
            
        # Entrenamiento del modelo
        model.fit(x_train,y_train)

        # Predicción del modelo ajustado para el conjunto de validación
        pred = model.predict(x_val)

        # Cálculo de los errores cuadráticos medios y Coeficiente de Determinación (R^2)
        mse_train = metrics.mean_squared_error(y_train, model.predict(x_train))
        R2_train = model.score(x_train, y_train)
        logging.info('Métricas del Modelo:')
        logging.info('ENTRENAMIENTO: RMSE: {:.2f} - R2: {:.4f}'.format(mse_train**0.5, R2_train))

        mse_val = metrics.mean_squared_error(y_val, pred)
        R2_val = model.score(x_val, y_val)
        logging.info('VALIDACIÓN: RMSE: {:.2f} - R2: {:.4f}'.format(mse_val**0.5, R2_val))

        logging.info('\nCoeficientes del Modelo:')
        # Constante del modelo
        logging.info('Intersección: {:.2f}'.format(model.intercept_))

        # Coeficientes del modelo
        coef = pd.DataFrame(x_train.columns, columns=['features'])
        coef['Coeficiente Estimados'] = model.coef_
        logging.info(coef)
        #coef.sort_values(by='Coeficiente Estimados').set_index('features').plot(kind='bar', title='Importancia de las variables', figsize=(12, 6))

        return model

    def model_dump(self, model_trained) -> None:
        import pickle
        """
        COMPLETAR DOCSTRING
        
        """
        
        # COMPLETAR CON CÓDIGO
        pickle.dump(model_trained,open(self.model_path+'model.sav','wb'))
        return None

    def run(self):
    
        df = self.read_data()
        model_trained = self.model_training(df)
        self.model_dump(model_trained)

if __name__ == "__main__":

    ModelTrainingPipeline(input_path = '../data/output/',
                          model_path = '../model/').run()