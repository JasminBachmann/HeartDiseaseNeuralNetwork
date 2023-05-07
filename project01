import os
import sys
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from google.colab import drive

#drive.mount('/content/drive')

# Get full path of notebook
#notebook_path = os.path.abspath("")

# Load data from CSV file
#data = pd.read_csv(os.path.join(notebook_path, "heart.csv"))
#data = pd.read_csv(os.path.join(sys.path[0], 'heart.csv'))

#data = pd.read_csv(drive/MyDrive/Colab Notebooks/'heart.csv')

#data = pd.read_csv("heart.csv", na_values='?')
#data = pd.read_csv('/content/drive/MyDrive/heart.csv')

url = "https://drive.google.com/file/d/13nkRXHuAXRtf3WxWmbggnJy0du90NAx4/view?usp=share_link"
names = ['id', 'age', 'sex', 'dataset', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
         'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num']
data = pd.read_csv(url, header=None, names=names, na_values='?')
print(data.head())

# Drop any rows with missing values
data.dropna(inplace=True)

# Split data into features and labels
X = data.drop('num',axis=1)
y = data.num
print(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the neural network
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=32, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(units=16, activation='relu'),
    tf.keras.layers.Dense(units=1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=32)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test accuracy:', test_acc)
