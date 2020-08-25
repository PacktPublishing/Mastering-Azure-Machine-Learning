import pandas as pd
from sklearn.preprocessing import StandardScaler
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
from azureml.core.run import Run
from keras.callbacks import Callback
import numpy as np
import argparse

run = Run.get_context()

parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type=int, dest='batch_size', default=50, help='mini batch size for training')
parser.add_argument('--epochs', type=int, dest='epochs', default=30, help='epochs')
parser.add_argument('--first-layer-neurons', type=int, dest='n_hidden_1', default=100,
                    help='# of neurons in the first layer')
parser.add_argument('--second-layer-neurons', type=int, dest='n_hidden_2', default=100,
                    help='# of neurons in the second layer')
parser.add_argument('--learning-rate', type=float, dest='learning_rate', default=0.01, help='learning rate')
parser.add_argument('--momentum', type=float, dest='momentum', default=0.9, help='momentum')
args = parser.parse_args()

class AzureMlKerasCallback(Callback):

    def __init__(self, run):
        super(AzureMlKerasCallback, self).__init__()
        self.run = run

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        send = {}
        send['epoch'] = epoch
        for k, v in logs.items():
            if isinstance(v, (np.ndarray, np.generic)):
                send[k] = v.item()
            else:
                send[k] = v
        for k, v in send.items():
            if isinstance(v, list):
                self.run.log_list(k, v)
            else:
                self.run.log(k, v)

df = pd.read_csv('train.csv')
df = df.drop_duplicates()

train = pd.get_dummies(df, columns=["Sex","Embarked","Cabin"], prefix=["Sex","Emb","Cabin"], drop_first=True)
train = train.fillna(0)
X_train = train.drop(["Survived","PassengerId","Name","Ticket"],axis=1).values
y_train = train["Survived"].values

X_train = StandardScaler().fit_transform(X_train)

model = Sequential()

model.add(Dense(args.n_hidden_1, activation='relu', input_dim=X_train.shape[1], kernel_initializer='uniform'))
model.add(Dropout(0.50))
model.add(Dense(args.n_hidden_2, kernel_initializer='uniform', activation='relu'))
model.add(Dropout(0.50))
model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))

sgd = SGD(lr = args.learning_rate, momentum = args.momentum)
model.compile(optimizer = sgd,  loss = 'binary_crossentropy',  metrics = ['accuracy'])

# Create an Azure Machine Learning monitor callback
azureml_cb = AzureMlKerasCallback(run)

model.fit(X_train, y_train, batch_size = args.batch_size, epochs = args.epochs, verbose=1, callbacks=[azureml_cb])

scores = model.evaluate(X_train, y_train, batch_size=30)
run.log(model.metrics_names[0], float(scores[0]))
run.log(model.metrics_names[1], float(scores[1]))