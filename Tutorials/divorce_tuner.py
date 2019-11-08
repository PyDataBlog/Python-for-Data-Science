# Import Libraries
import tensorflow
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from rarfile import RarFile
from urllib.request import urlretrieve
from kerastuner import HyperModel
from kerastuner.tuners import RandomSearch, Hyperband, BayesianOptimization
from tensorflow import keras
from sklearn.metrics import confusion_matrix


sns.set(rc={'figure.figsize': (20, 10)})

# download the compressed divorce file
urlretrieve('https://archive.ics.uci.edu/ml/machine-learning-databases/00497/divorce.rar', 'divorce_file.rar')

# extract rar file
with RarFile('divorce_file.rar', mode='r') as rf:
  rf.extractall()

# read divorce data
df = pd.read_excel('divorce.xlsx')

# clean columns
clean_cols = [x.lower() for x in df.columns.to_list()]
df.columns = clean_cols

# Separate the target and features as separate dataframes
X = df.drop('class', axis=1)
y = df[['class']].astype('int')

# Stratified split based on the distribution of the target vector, y
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    stratify=y,
                                                    test_size=0.20,
                                                    random_state=30)


class MyHyperModel(HyperModel):

    def __init__(self, num_classes):
        self.num_classes = num_classes

    def build(self, hp):
        
        # specify model
        model = keras.Sequential()

        # range of models to build
        for i in range(hp.Int('num_layers', 2, 20)):

            model.add(keras.layers.Dense(units=hp.Int('units_' + str(i),
                                                min_value=32,
                                                max_value=512, 
                                                step=32),
                                   activation='relu'))

        model.add(keras.layers.Dense(self.num_classes, activation='sigmoid'))
        
        model.compile(
            optimizer=keras.optimizers.Adam(
                hp.Choice('learning_rate',
                          values=[1e-2, 1e-3, 1e-4])),
            loss='binary_crossentropy',
            metrics=['accuracy'])

        return model
    


hypermodel = MyHyperModel(num_classes=1)    

tuner = Hyperband(
    hypermodel,
    objective='accuracy',
    max_epochs=10,
    seed=10,
    project_name='divorce test')


tuner.search(X_train.values, y_train.values.flatten(),
             epochs=10,
             validation_data=(X_test.values, y_test.values.flatten()))

params = tuner.get_best_hyperparameters()[0]

model = tuner.hypermodel.build(params)

model.fit(X.values, y.values.flatten(), epochs=20)

hyperband_accuracy_df = pd.DataFrame(model.history.history)

hyperband_accuracy_df[['loss', 'accuracy']].plot()
plt.title('Loss & Accuracy Per EPOCH')
plt.xlabel('EPOCH')
plt.ylabel('Accruacy')
plt.show()


random_tuner = RandomSearch(
    hypermodel,
    objective='accuracy',
    max_trials=10,
    seed=10, 
    project_name='divorce test')


random_tuner.search(X_train.values, y_train.values.flatten(),
             epochs=10,
             validation_data=(X_test.values, y_test.values.flatten()))

random_params = random_tuner.get_best_hyperparameters()[0]

random_model = random_tuner.hypermodel.build(params)

random_model.fit(X.values, y.values.flatten(), epochs=15)

random_accuracy_df = pd.DataFrame(random_model.history.history)

random_accuracy_df[['loss', 'accuracy']].plot()
plt.title('Loss & Accuracy Per EPOCH For Random Model')
plt.xlabel('EPOCH')
plt.ylabel('Accruacy')
plt.show()



bayesian_tuner = BayesianOptimization(
    hypermodel,
    objective='accuracy',
    max_trials=10,
    seed=10,
    project_name='divorce test')

bayesian_tuner.search(X_train.values, y_train.values.flatten(),
             epochs=10,
             validation_data=(X_test.values, y_test.values.flatten()))
             
bayesian_params = bayesian_tuner.get_best_hyperparameters()[0]

bayesian_model = bayesian_tuner.hypermodel.build(bayesian_params)

bayesian_model.fit(X.values, y.values.flatten(), epochs=15)

bayesian_accuracy_df = pd.DataFrame(bayesian_model.history.history)

bayesian_accuracy_df[['loss', 'accuracy']].plot()
plt.title('Loss & Accuracy Per EPOCH For Bayesian Optimisation Model')
plt.xlabel('EPOCH')
plt.ylabel('Accruacy')
plt.show()




