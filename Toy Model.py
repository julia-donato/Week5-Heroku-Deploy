import pandas as pd
import pickle 
from sklearn.model_selection import train_test_split
# Read Data

df = pd.read_csv('USA_Housing.csv')

# Select independent and dependent variables

X = df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
       'Avg. Area Number of Bedrooms', 'Area Population',]]

y = df['Price']

# Split data into train and test

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)

# Training the Model

from sklearn.linear_model import LinearRegression

lm = LinearRegression()

lm.fit(X_train,y_train)

# Make pickle file

pickle.dump(lm, open("model.pkl","wb"))



