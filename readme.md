

# TP Machine Learning II

***This project is for the Machine Learning II class***

## Overview

The following flowcharts represents how this project is divided:

``` mermaid
flowchart 
   subgraph training pipeline
   data[(csv files)]  --> feature_engineering.py --> features
   features --> train.py --> model.pkl
   end
``` 

``` mermaid
flowchart 
   subgraph inference pipeline
   data[(csv/json files)]  --> feature_engineering.py --> features
   model.pkl --> predict.py 
   features --> predict.py --> predictions
   end
```
# Training pipeline

To run the training pipeline, just execute the following in the terminal

``` python
python train_pipeline.py 
```

# Inference pipeline

To test the train model, just execute the following:
``` python
python inference_pipeline.py
```

## **NOTE**

By default the inference_pipeline.py script uses the ***example.json*** file, to showcase the product's capability of handling json files as this type of files would be commonly used if the model gets deployed onto an on-line or on-demand model by using an API.

# Testing individual scripts

You can check which parameters every script expects by running: 

```python
python feature_engineering.py -h
```

## Feature engineering script

``` python
python feature_engineering.py -i ../data/ -o ../data/output/output_name
```

## Train script

``` python
python train.py -i ../data/output/ -o ../model
```

## Predict script


### For **csv** files
``` python
python predict.py -i ../data/output/test_final.csv  -o ../model/output/ -m ../model/
```

### For **json** files
``` python
python predict.py -i ../Notebook/example.json  -o ../model/output/ -m ../model/
```

# Pylint Scores:

   - train.py 10/10
   - predict.py 8.75/10 (due to helper module import. but code runs)
   - feature_engineering.py 9.51/10 (Broad Exceptions)
# Challenges

The major challenges I had with this project were the following:

- Managing to feed json files

For that a utils.py module was created under the src/helpers folder
a  helper function called generic_reader was implemented, which wraps 3 pandas read_ functions, being, parquet, csv, and json, which means the product can handle those three file types.

- Writing json files

Same as before, another helper function was develop, but this one can actually handle all exporting types supported by pandas.

- Hyperparameter optimization

Using optuna the following models were used:

ridge, lasso, elasticnet, decision trees, random forest, gradientBoost, and a small neural network 

But the r2 score of all models converge to a range between 0.58-0.59, by altering hyperameters, no noticeable improvements were seen.


