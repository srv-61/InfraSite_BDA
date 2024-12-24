import os

path = r'D:\SOHAN\7TH SEM\Big Data & Deep Learning\InfraSite_BDA\Datasets\Train Datasets'
path = r'D:\SOHAN\7TH SEM\Big Data & Deep Learning\InfraSite_BDA\Datasets\Validation Datasets'

if os.path.exists(path):
    print("Path exists!")
else:
    print("Path does not exist.")
