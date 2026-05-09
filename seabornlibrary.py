import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Loading the dataset
df = pd.read_csv("https://drive.google.com/file/d/1CwY0n46V3kGVehhLePTO69hXd7abw9nN/view?usp=sharing")

#Displaying the first 10 rows
print(df.head(10))

#Getting dataset information
print(df.info())

#Getting statistical summary
print(df.describe())

print(df.columns)

sns.pairplot(df.select_dtypes(include=[np.number]))

sns.heatmap(df.select_dtypes(include=[np.number]).corr(),annot = True)

plt.show()
