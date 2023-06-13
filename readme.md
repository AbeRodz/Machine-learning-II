

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
   data[(csv files)]  --> feature_engineering.py --> features
   model.pkl --> features
   features --> predict.py --> predictions
   end
```
## Training pipeline

To run the training pipeline, just execute the following in the terminal

``` python
python feature_engineering.py -i "input/dataset/path" -o "output/dataset/path"

```

You can check the arguments by running:

``` python
python feature_engineering.py -h

```
