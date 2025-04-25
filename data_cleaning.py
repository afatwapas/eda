import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from scipy.stats import zscore
df=pd.read_csv('train.csv')
print(df.head())
print(df.shape)
print(df.info())
print(df.describe())

print(df.columns)
#   PREPROCESSING THE DATA, PERFORMING ONE-HOT ENCODIN ON CATEGORICAL DATA,DROPPING UNNECESSARY DATA , FILLING MISSING DATA

# One-hot encode the 'sex' column, drop the first category to avoid dummy variable trap

Male = pd.get_dummies(df['Sex'], prefix='Sex', drop_first=True)

# Concatenate the new columns with the original DataFrame
df = pd.concat([df, Male], axis=1)

# Drop the original 'sex' column
df.drop('Sex', axis=1, inplace=True)



# One-hot encode the 'embarkd' column, drop the first category to avoid dummy variable trap

embarked = pd.get_dummies(df['Embarked'], prefix='Embarked', drop_first=True)

# Concatenate the new columns with the original DataFrame
df = pd.concat([df, embarked], axis=1)

# Drop the original 'embarked' column
df.drop('Embarked', axis=1, inplace=True)
df.drop('Cabin',axis=1,inplace=True)
df.drop('Ticket',axis=1,inplace=True)
df['Age']= df['Age'].fillna(df['Age'].mean())

df= df.replace({True:1,False:0})

#PLOTTING THE DATA

#graph of survived column
sns.countplot(x='Survived',data=df)
plt.title('Distribution of data based on output variable')
plt.show()

#plot of people survived and their gender
sns.countplot(data=df, x='Survived', hue='Sex_male')
plt.title('Survival Count by Sex')
plt.xlabel('Survived (0 = No, 1 = Yes)')
plt.ylabel('Number of Passengers')
plt.legend(title='Sex')
plt.show()

#OUTLIER DETECTION
#OUTLIER DETECTION ON COLUMN "AGE"
sns.boxplot(x=df['Age'])
plt.title('boxplot of "age" before outlier removal')
plt.show()
#OUTLIER DETECTION ON COLUMN "Fare"
sns.boxplot(x=df['Fare'])
plt.title('boxplot of "fare" before outlier removal')
plt.show()
#outlier removal
column='Fare'
Q1 = df[column].quantile(0.25)
Q3 = df[column].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df_cleaned = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

#boxplot after outlier removal
sns.boxplot(x=df_cleaned['Fare'])
plt.title('boxplot of "fare" after outlier removal')
plt.show()

#SCALING THE NUMERICAL FEATURES
scaler= StandardScaler()
df_cleaned[['Age','Fare']]= scaler.fit_transform(df_cleaned[['Age','Fare']])

#HEATMAP to correlate features
numeric_df=df_cleaned.select_dtypes(include=['number'])
corr= numeric_df.corr()
plt.figure(figsize=(10, 8))  
sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5, fmt='.2f')
plt.show()

#ANOMALY DETECTION USING Z-SCORE
numeric_df['z_score'] = zscore(numeric_df['Fare'])
outliers = numeric_df[numeric_df['z_score'].abs() > 3]
print(outliers)
