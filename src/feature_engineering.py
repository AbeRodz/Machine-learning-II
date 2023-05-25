"""
feature_engineering.py

COMPLETAR DOCSTRING

DESCRIPCIÓN: Feature Engineering Pipeline Scrpt para modelo 
AUTOR: Abraham Rodriguez
FECHA: 24/5/2023
"""

# Imports
import pandas as pd
import numpy as np
class FeatureEngineeringPipeline(object):

    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path

    def read_data(self) -> pd.DataFrame:
        """
        Function that reads from data from a the ./data dir, and performs
        merge.
        
        :return pandas_df: The desired DataLake table as a DataFrame
        :rtype: pd.DataFrame
        """
            
        # COMPLETAR CON CÓDIGO
        data_train = pd.read_csv('../data/Train_BigMart.csv')
        data_test = pd.read_csv('../data/Test_BigMart.csv')
        # Identificando la data de train y de test, para posteriormente unión y separación
        data_train['Set'] = 'train'
        data_test['Set'] = 'test'
        pandas_df = pd.concat([data_train, data_test], ignore_index=True, sort=False)
       
        return pandas_df

    
    def data_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        COMPLETAR DOCSTRING
        Performs data cleaning and feature engineering on the dataset.

        :return df_transformed: Transformed dataset
        :rtype: pd.DataFrame
        """
        
        # COMPLETAR CON CÓDIGO
        data = df
        data['Outlet_Establishment_Year'] = 2020 - data['Outlet_Establishment_Year']
        data['Item_Fat_Content'] = data['Item_Fat_Content'].replace({'low fat':  'Low Fat', 'LF': 'Low Fat', 'reg': 'Regular'})
        
        productos = list(data[data['Item_Weight'].isnull()]['Item_Identifier'].unique())
        
        for producto in productos:  
            moda = (data[data['Item_Identifier'] == producto][['Item_Weight']]).mode().iloc[0,0]
            data.loc[data['Item_Identifier'] == producto, 'Item_Weight'] = moda
        
        
        outlets = list(data[data['Outlet_Size'].isnull()]['Outlet_Identifier'].unique())
        
        for outlet in outlets:
            data.loc[data['Outlet_Identifier'] == outlet, 'Outlet_Size'] =  'Small'
        
        
        data.loc[data['Item_Type'] == 'Household', 'Item_Fat_Content'] = 'NA'
        data.loc[data['Item_Type'] == 'Health and Hygiene', 'Item_Fat_Content'] = 'NA'
        data.loc[data['Item_Type'] == 'Hard Drinks', 'Item_Fat_Content'] = 'NA'
        data.loc[data['Item_Type'] == 'Soft Drinks', 'Item_Fat_Content'] = 'NA'
        data.loc[data['Item_Type'] == 'Fruits and Vegetables', 'Item_Fat_Content'] = 'NA'
        # FEATURES ENGINEERING: creando categorías para 'Item_Type'
        data['Item_Type'] = data['Item_Type'].replace({'Others': 'Non perishable', 'Health and Hygiene': 'Non perishable', 'Household': 'Non perishable',
        'Seafood': 'Meats', 'Meat': 'Meats',
        'Baking Goods': 'Processed Foods', 'Frozen Foods': 'Processed Foods', 'Canned': 'Processed Foods', 'Snack Foods': 'Processed Foods',
        'Breads': 'Starchy Foods', 'Breakfast': 'Starchy Foods',
        'Soft Drinks': 'Drinks', 'Hard Drinks': 'Drinks', 'Dairy': 'Drinks'})

        # FEATURES ENGINEERING: asignación de nueva categorías para 'Item_Fat_Content'
        data.loc[data['Item_Type'] == 'Non perishable', 'Item_Fat_Content'] = 'NA'
        
        data['Item_MRP'] = pd.qcut(data['Item_MRP'], 4, labels = [1, 2, 3, 4])
        dataframe = data.drop(columns=['Item_Type', 'Item_Fat_Content']).copy()
        serie_var = dataframe['Outlet_Size'].unique()
        serie_var.sort()
        print('Outlet_Size', ':', serie_var)

        serie_var = dataframe['Outlet_Location_Type'].unique()
        serie_var.sort()
        print('Outlet_Location_Type', ':', serie_var)

        # Codificación de variables ordinales
        dataframe['Outlet_Size'] = dataframe['Outlet_Size'].replace({'High': 2, 'Medium': 1, 'Small': 0})
        dataframe['Outlet_Location_Type'] = dataframe['Outlet_Location_Type'].replace({'Tier 1': 2, 'Tier 2': 1, 'Tier 3': 0}) # Estas categorias se ordenaron asumiendo la categoria 2 como más lejos
        df_transformed = pd.get_dummies(dataframe, columns=['Outlet_Type'])  

        return df_transformed

    def write_prepared_data(self, transformed_dataframe: pd.DataFrame) -> None:
        """
        COMPLETAR DOCSTRING
        Writes onto the output_path directory the transformed dataset created by the data_transformation function
        
        """
        
        # COMPLETAR CON CÓDIGO
        # Eliminación de variables que no contribuyen a la predicción por ser muy específicas
        dataset = transformed_dataframe.drop(columns=['Item_Identifier', 'Outlet_Identifier'])

        # División del dataset de train y test
        df_train = dataset.loc[dataset['Set'] == 'train']
        df_test = dataset.loc[dataset['Set'] == 'test']

        # Eliminando columnas sin datos
        df_train.drop(['Set'], axis=1, inplace=True)
        df_test.drop(['Item_Outlet_Sales','Set'], axis=1, inplace=True)

        # Guardando los datasets
        df_train.to_csv(self.output_path+"/train_final.csv")
        df_test.to_csv(self.output_path+"/test_final.csv")
        return None

    def run(self):
    
        df = self.read_data()
        df_transformed = self.data_transformation(df)
        self.write_prepared_data(df_transformed)

  
if __name__ == "__main__":
    FeatureEngineeringPipeline(input_path = '../data/',
                               output_path = '../data/output').run()