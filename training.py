from sklearn.model_selection import train_test_split # Allows us to train on one partition, and then test it on another one
from tensorflow.keras.utils import to_categorical 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
import numpy as np
import os

class ML_Model():
    def __init__(self):
        self.actions = np.array(['ayuda', 'clase', 'donde', 'gracias', 'hola', 'necesitar', 'no_entender', 'repetir', 'n-a', 'empty'])
        self.label_map = {label:num for num, label in enumerate(actions)}
        self.no_sequences = 200
        self.sequence_length = 30
        self.DATA_PATH = "/Users/codingdan/Documents/University/Semestre 2 2021-2022/Tesina/codigo/ML_model training/MP_Data"
        self.sequences, self.labels = [],[]
        self.structure_data()
        # Turn the labels into a categorization list. This turns the labels list into a list of list, where each sub-list contains a set of numbers indicating the category
        self.labels_categorized = to_categorical(self.labels).astype(int) 

    def structure_data(self):
        sequences = []
        for action in self.actions:
            for sequence in range(self.no_sequences):
                window = []
                for frame_num in range(self.sequence_length):
                    res = np.load(os.path.join(self.DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
                    window.append(res)
                sequences.append(window)
                self.labels.append(self.label_map[action])

        self.sequences = np.array(sequences) # turn the list into a numpy array
        