import subprocess
subprocess.run(['Python', 'feature_engineering.py' ,'-i ../Notebook/example.json' ,'-o ../data/output/feature_engineering'])

# uncomment to run with test_final.csv
# subprocess.run(['Python', 'predict.py','-i ../data/output/test_final.csv',
#                  '-o ../model/output/', '-m ../model/'])

# run to test with generated csv from example.json 
subprocess.run(['Python', 'predict.py','-i ../data/output/feature_engineering_example.csv',
                 '-o ../model/output/predictions_inference_example', '-m ../model/'])