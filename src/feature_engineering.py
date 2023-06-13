"""
feature_engineering.py

COMPLETAR DOCSTRING

DESCRIPCIÓN: Feature Engineering Pipeline Scrpt para modelo
AUTOR: Abraham Rodriguez
FECHA: 24/5/2023
"""

# Imports
import os
import logging
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s',
    datefmt='%H:%M:%S')


class FeatureEngineeringPipeline():
    """
    Class to perform feature engineering and cleaning to a dataset.

    """

    def __init__(self, input_path: str, output_path: str) -> None:
        """
        Class constructor, takes the input path (files to read), and the output path, to
        write on disk the transformed dataset.

        :param input_path : input directory path
        :type path : string

        :param output_path : output directory path
        :type path : string

        """
        self.input_path = input_path
        self.output_path = output_path

    @staticmethod
    def generic_reader(path: str) -> pd.DataFrame:
        """
        Generic function that handles reading parquet, csv, and json files
        returns a DataFrame.

        :param path : filepath
        :type path : string

        :return type : pd.DataFrame

        """

        format_type = path.split('.')[-1]
        try:
            func = getattr(pd, f'read_{format_type}')
            if format_type == 'parquet':
                engine = 'auto'
                return func(path, engine)

            if format_type == 'csv':
                return func(path)

            if format_type == 'json':
                return func(path, lines=True)
        except Exception as err:
            raise TypeError("format not handled") from err

    def read_data(self) -> pd.DataFrame | None:
        """
        Function that finds test and train datasets from the ./data dir, and performs
        merge.

        :return type: pd.DataFrame

        """

        items = os.listdir(self.input_path)

        for dataset in items:
            try:
                if dataset.lower().startswith('test'):
                    test_path = dataset
                if dataset.lower().startswith('train'):
                    train_path = dataset
            except LookupError("couldn't find test and train files") as err:
                logging.error(err)
                return None
        try:
            data_train = self.generic_reader(self.input_path + train_path)

            data_test = self.generic_reader(self.input_path + test_path)

            data_train['Set'] = 'train'
            data_test['Set'] = 'test'
            pandas_df = pd.concat([data_train, data_test],
                                  ignore_index=True, sort=False)
            return pandas_df
        except TypeError as err:
            logging.error(err)
            return None

    @staticmethod
    def replace_column_with_value(data_frame: pd.DataFrame,
                                  column: str,
                                  current_value: str,
                                  target: str,
                                  value) -> None:
        """
        Function that replaces a value from a column, by considerating a filter.

        :param data_frame : target dataframe
        :type data_frame : pd.DataFrame

        :param column : target column
        :type column : str

        :param filter : filtering value
        :type filter : str

        :param target : current target value
        :type target : str

        :param value : new value
        :type value : any

        """
        data_frame.loc[data_frame[column] == current_value, target] = value

    def clean_frame(self, data_frame):
        """
        Performs all data cleaning procedures.

        :param data_frame : target dataframe
        :type data_frame : pd.DataFrame

        """
        data_frame['Item_Fat_Content'] = data_frame['Item_Fat_Content'].replace(
            {'low fat': 'Low Fat', 'LF': 'Low Fat', 'reg': 'Regular'})

        productos = list(data_frame[data_frame['Item_Weight'].isnull()]
                         ['Item_Identifier'].unique())

        for producto in productos:
            moda = (data_frame[data_frame['Item_Identifier'] == producto]
                    [['Item_Weight']]).mode().iloc[0, 0]
            self.replace_column_with_value(
                data_frame=data_frame,
                column='Item_Identifier',
                current_value=producto,
                target='Item_Weight',
                value=moda)
            # data_frame.loc[data_frame['Item_Identifier']
            #               == producto, 'Item_Weight'] = moda

        outlets = list(data_frame[data_frame['Outlet_Size'].isnull()]
                       ['Outlet_Identifier'].unique())

        for outlet in outlets:
            self.replace_column_with_value(
                data_frame=data_frame,
                column='Outlet_Identifier',
                current_value=outlet,
                target='Outlet_Size',
                value='Small')
            # data_frame.loc[data_frame['Outlet_Identifier'] ==
            #                outlet, 'Outlet_Size'] = 'Small'

    def data_transformation(self, data_frame: pd.DataFrame) -> pd.DataFrame:
        """
        Performs data cleaning and feature engineering on the dataset.

        :param data_frame : Target DataFrame
        :type path : pd.DataFrame

        :return type : pd.DataFrame
        """

        # feature
        data_frame['Outlet_Establishment_Year'] = 2020 - \
            data_frame['Outlet_Establishment_Year']

        self.clean_frame(data_frame)
        # feature
        item_types = [
            'Household',
            'Health and Hygiene',
            'Hard Drinks',
            'Soft Drinks',
            'Fruits and Vegetables']
        for i in item_types:
            data_frame.loc[data_frame['Item_Type']
                           == i, 'Item_Fat_Content'] = 'NA'
         # feature
        item_types_categories = [
            {'Non perishable': ['Others', 'Health and Hygiene', 'Household']},
            {'Meats': ['Seafood', 'Meat']},
            {'Processed Foods': ['Baking Goods', 'Frozen Foods', 'Canned', 'Snack Foods']},
            {'Starchy Foods': ['Breads', 'Breakfast']},
            {'Drinks': ['Soft Drinks', 'Hard Drinks', 'Dairy']}
        ]

        for i in item_types_categories:
            for key, value in i.items():
                for j in value:

                    data_frame['Item_Type'] = data_frame['Item_Type'].replace(
                        j, key)

        # FEATURES ENGINEERING: asignación de nueva categorías para
        # 'Item_Fat_Content'
        data_frame.loc[data_frame['Item_Type'] == 'Non perishable',
                       'Item_Fat_Content'] = 'NA'
        # feature
        data_frame['Item_MRP'] = pd.qcut(
            data_frame['Item_MRP'], 4, labels=[1, 2, 3, 4])

        dataframe = data_frame.drop(
            columns=[
                'Item_Type',
                'Item_Fat_Content']).copy()

        # feature
        # Codificación de variables ordinales
        dataframe['Outlet_Size'] = dataframe['Outlet_Size'].replace(
            {'High': 2, 'Medium': 1, 'Small': 0})
        dataframe['Outlet_Location_Type'] = dataframe['Outlet_Location_Type'].replace(
            {'Tier 1': 2, 'Tier 2': 1, 'Tier 3': 0})
        # feature
        df_transformed = pd.get_dummies(dataframe, columns=['Outlet_Type'])

        return df_transformed

    def write_prepared_data(self, transformed_dataframe: pd.DataFrame) -> None:
        """
        Writes onto the output_path directory the transformed dataset
        created by the data_transformation function.

        :param transformed_dataframe : Target DataFrame
        :type path : pd.DataFrame

        """

        dataset = transformed_dataframe.drop(
            columns=['Item_Identifier', 'Outlet_Identifier'])

        df_train = dataset.loc[dataset['Set'] == 'train']
        df_test = dataset.loc[dataset['Set'] == 'test']

        df_train = df_train.drop(['Set'], axis=1)
        df_test = df_test.drop(['Item_Outlet_Sales', 'Set'], axis=1)

        df_train.to_csv(self.output_path + "/train_final.csv")
        df_test.to_csv(self.output_path + "/test_final.csv")

    def run(self):
        """
        Executes the feature engineering pipeline, performs data reading, transformation
        and writes onto disk the transformed data.
        """

        logging.info('Reading data...')

        data_frame = self.read_data()
        if data_frame is None:
            return

        logging.info('Data cleaning and transformation...')

        df_transformed = self.data_transformation(data_frame)

        logging.info('Writing data...')

        self.write_prepared_data(df_transformed)

        logging.info('Process finished')


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description="Script that executes data cleaning and transformation,\
          generates training and test datasets onto the specified output path")

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
    if args.output_path is None:
        args.output_path = "../data/output/"
    FeatureEngineeringPipeline(input_path=args.input_path,
                               output_path=args.output_path).run()
