import os
import pandas as pd

dataset_path = "./mbti_1.csv"
df = pd.read_csv(dataset_path)

print(df['type'].unique()) 
print(df['type'].value_counts())
