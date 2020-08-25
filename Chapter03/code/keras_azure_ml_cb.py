from keras.callbacks import Callback
import numpy as np

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
