"""
predict.py

COMPLETAR DOCSTRING

DESCRIPCIÓN: Script de prediccion, que utiliza el modelo generado
AUTOR: Abraham Rodriguez
FECHA: 24/5/2023
"""

# Imports

import logging
import pickle
import pandas as pd

from feature_engineering import FeatureEngineeringPipeline

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s',
    datefmt='%H:%M:%S')


class MakePredictionPipeline(FeatureEngineeringPipeline):
    """
    Class to perform model inference onto a dataset.

    """
    def __init__(self, input_path, output_path, model_path: str = None):
        """
        Class constructor, takes the input path (files to read), and the output path, to
        write on disk the model predicions, uses a pre-saved model.

        :param input_path : input directory path
        :type input_path : string

        :param output_path : output directory path
        :type output_path : string
        
        :param model_path :  saved model path
        :type model_path : string
        """
        self.input_path = input_path
        self.output_path = output_path
        self.model_path = model_path
        self.model = None

    def load_data(self) -> pd.DataFrame:
        """
        COMPLETAR DOCSTRING
        """

        data = self.generic_reader(self.input_path + 'test_final.csv', index_col=[0])
        # data = pd.read_csv(self.input_path + 'test_final.csv',index_col=0)

        return data

    def load_model(self) -> None:
        """
        Loads a pre-saved model using the pickle module

        """

        with open(self.model_path +"model.sav",'rb') as saved_model:
            self.model = pickle.load(saved_model)

    def make_predictions(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Performs predictions using .predict function, by calling the
        pre-saved model

        :param data : dataframe that contains data to evaluate.
        :type data : pd.DataFrame

        """

        predictions = self.model.predict(data)

        return pd.DataFrame(predictions)

    def write_predictions(self, predicted_data: pd.DataFrame) -> None:
        """
        Writes a csv file of predicted values from the model.

        :param predicted_data : dataframe that contains predictions.
        :type predicted_data : pd.DataFrame

        """
        print(predicted_data)
        predicted_data.to_csv(self.output_path + 'data_test.csv')

    def run(self):
        """
        Executes the inference pipeline, loads data, and the model,
        makes and writes predictions onto a .csv file

        """
        logging.info('Reading data...')

        data = self.load_data()
        print(data.head())
        logging.info('Loading model...')

        self.load_model()

        logging.info('Making predictions...')

        df_preds = self.make_predictions(data)

        logging.info('Writing data...')
        self.write_predictions(df_preds)


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description="Script that executes inference from a pre-saved model,\
          generates predictions onto a .csv file")

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
    parser.add_argument(
        "-m",
        "--model_path",
        type=str,
        help="saved model path",
        required=True
    )
    args = parser.parse_args()
    # spark = Spark()

    # pipeline = MakePredictionPipeline(input_path = '../data/output/',
    #                                   output_path = '../model/output/',
    #                                   model_path = '../model/')
    pipeline = MakePredictionPipeline(input_path=args.input_path,
                                      output_path=args.output_path,
                                      model_path=args.model_path)
    pipeline.run()
