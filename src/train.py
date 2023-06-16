"""
train.py

COMPLETAR DOCSTRING

DESCRIPCIÓN:
AUTOR:
FECHA:
"""

# Imports
import logging
import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LinearRegression


logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s',
    datefmt='%H:%M:%S')


class ModelTrainingPipeline():
    """
    Class to perform model training using a dataset.

    """

    def __init__(self, input_path, model_path):
        """
        Class constructor, takes the input path (files to read), and the output path, to
        write on disk the transformed dataset.

        :param input_path : input directory path
        :type input_path : string

        :param output_path : output directory path
        :type output_path : string
        """
        self.input_path = input_path
        self.model_path = model_path

    def read_data(self) -> pd.DataFrame:
        """
        Reads data from the specified input_path.

        :return Dataframe 
        :rtype: pd.DataFrame
        """

        items = os.listdir(self.input_path)

        for dataset in items:
            if dataset.lower().startswith('train'):
                train_path = dataset

        data_train = pd.read_csv(self.input_path + train_path, index_col=0)

        return data_train

    def model_training(self, data_frame: pd.DataFrame) -> pd.DataFrame:
        """
        Performs model training with a selected dataset, and creates
        a new dataframe with predicted values

        :param data_frame : Target DataFrame
        :type data_frame : pd.DataFrame

        :return type : pd.DataFrame
        """
        seed = 28
        model = LinearRegression()

        # División de dataset de entrenaimento y validación

        x_data = data_frame.drop(columns='Item_Outlet_Sales')
        x_train, x_val, y_train, y_val = train_test_split(
            x_data, data_frame['Item_Outlet_Sales'], test_size=0.3, random_state=seed)

        # Entrenamiento del modelo
        model.fit(x_train, y_train)

        # Predicción del modelo ajustado para el conjunto de validación
        pred = model.predict(x_val)

        # Cálculo de los errores cuadráticos medios y Coeficiente de
        # Determinación (R^2)
        mse_train = metrics.mean_squared_error(y_train, model.predict(x_train))
        r2_train = model.score(x_train, y_train)
        logging.info('Model Metrics:')
        logging.info('TRAINING: RMSE: %f - R2: %f', mse_train**0.5 ,r2_train)

        mse_val = metrics.mean_squared_error(y_val, pred)
        r2_val = model.score(x_val, y_val)
        logging.info('VALIDATION: RMSE: %f - R2: %f' ,mse_val**0.5  ,r2_val)

        logging.info('\n Model Coefficients:')
        # Constante del modelo
        logging.info('\nIntersection: %s', model.intercept_)

        # Coeficientes del modelo
        coef = pd.DataFrame(x_train.columns, columns=['features'])
        coef['Estimated Coefficients'] = model.coef_
        logging.info('\n %s' ,coef)


        return model

    def model_dump(self, model_trained: LinearRegression) -> None:
        """
        Dumps the model onto a .pkl file into the directory declared on model_path
        :param model_trained : trained model.
        :type : LinearRegression

        """
        with open(self.model_path + 'model.pkl', 'wb') as model:
            pickle.dump(model_trained, model)

    def run(self):
        """
        Executes the training pipeline, performs data reading, training
        and writes onto disk the trained model.
        """

        logging.info('Reading data...')
        data_frame = self.read_data()

        logging.info('Training model...')
        model_trained = self.model_training(data_frame)

        logging.info('Exporting model...')
        self.model_dump(model_trained)
        logging.info('Model saved on %s directory',self.model_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Script that executes model training,and saves it onto the specified output path")

    parser.add_argument(
        "-i",
        "--input_path",
        type=str,
        help="input file path",
        required=True
    )
    parser.add_argument(
        "-o",
        "--output_path",
        type=str,
        help="output file path",
        required=False
    )

    args = parser.parse_args()
    ModelTrainingPipeline(input_path='../data/output/',
                          model_path='../model/').run()
