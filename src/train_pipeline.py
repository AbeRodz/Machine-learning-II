import subprocess

subprocess.run(['Python', 'feature_engineering.py', '-i ../data/' ,'-o ../data/output/'])

subprocess.run(['Python', 'train.py', '-i ../data/output/', '-o ../model/'])