import numpy as np
from sklearn.datasets import fetch_california_housing
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from keras import layers

# Data loading and preprocessing
california_housing_data = fetch_california_housing(as_frame=True)
train_split = 0.7
train_size = int(len(california_housing_data.data) * train_split)
train = california_housing_data.frame[:train_size]
test = california_housing_data.frame[train_size:]

X_train = train.drop(columns='MedHouseVal')
y_train = train['MedHouseVal']
X_test = test.drop(columns='MedHouseVal')
y_test = test['MedHouseVal']

def plot_data(latitude, longtitude, median_house_value, title, xlabel, ylabel):
    plt.figure(figsize=(10, 10))
    plt.scatter(longtitude, latitude, c=median_house_value, cmap='viridis')
    plt.colorbar()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
    

def plot_correlation_matrix(data):
    plt.figure(figsize=(10, 10))
    sns.heatmap(data.corr(), annot=True, cmap='viridis')
    plt.title("Correlation Matrix")
    plt.show()
    
plot_correlation_matrix(california_housing_data.frame)
plot_data(california_housing_data.frame['Latitude'], california_housing_data.frame['Longitude'], california_housing_data.frame['MedHouseVal'], 'Median House Value', 'Latitude', 'Longitude')
# Model Class
class CaliforniaHousingModel(keras.Model):
    def __init__(self):
        super(CaliforniaHousingModel, self).__init__()
        self.dense1 = layers.Dense(64, activation='relu', input_shape=[8])
        self.dense2 = layers.Dense(64, activation='relu')
        self.out_layer = layers.Dense(1)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.out_layer(x)

# Training Loop Function
def train_model(model, X_train, y_train, X_test, y_test, epochs=10, batch_size=32):
    model.compile(optimizer='adam', loss='mean_squared_error') 

    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(patience=3), 
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True) 
    ]

    history = model.fit(X_train, y_train, 
                        epochs=epochs, 
                        batch_size=batch_size, 
                        validation_data=(X_test, y_test),
                        callbacks=callbacks)
    return history

# Model Creation and Training
model = CaliforniaHousingModel()
history = train_model(model, X_train, y_train, X_test, y_test)

# Evaluation
model.evaluate(X_test, y_test)

# Predictions
predictions = model.predict(X_test)

# Plotting
plt.figure(figsize=(10, 10))
plt.subplot(1, 2, 1)
plt.scatter(X_test['Longitude'], X_test['Latitude'], c=y_test, cmap='viridis')
plt.colorbar()
plt.title('Ground Truth')
plt.xlabel('Longitude')
plt.ylabel('Latitude')

plt.subplot(1, 2, 2)
plt.scatter(X_test['Longitude'], X_test['Latitude'], c=predictions, cmap='viridis')
plt.colorbar()
plt.title('Predictions')
plt.xlabel('Longitude')
plt.ylabel('Latitude')

plt.tight_layout()
plt.show()
