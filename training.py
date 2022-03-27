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
        self.label_map = {label:num for num, label in enumerate(self.actions)}

        self.no_sequences = 200
        self.sequence_length = 30
        self.DATA_PATH = "/Users/codingdan/Documents/University/Semestre 2 2021-2022/Tesina/codigo/ML_model training/MP_Data"

        self.sequences, self.labels = [],[]
        self.structure_data()
        # Turn the labels into a categorization list. This turns the labels list into a list of list, where each sub-list contains a set of numbers indicating the category
        self.labels_categorized = to_categorical(self.labels).astype(int) 

        self.log_dir = os.path.join('Logs')
        #TensorBoard is a web app that comes with tensorflow that allows you to monitor the neural network training
        # Monitor training and accuracy as it's being trained
        self.tb_callback = TensorBoard(log_dir=self.log_dir)

        self.model = self.build_neural_network_architecture()

    def structure_data(self):
        """ Goes action by action and adds all collected sequence data to the sequences list
            for each sequence added to sequences, its corresponding label is added to self.labels
        """
        print("NOW STRUCTURING SEQUENCE DATA")
        sequences = []
        for action in self.actions:
            print(f"Adding {action} data")
            for sequence in range(self.no_sequences):
                window = []
                for frame_num in range(self.sequence_length):
                    res = np.load(os.path.join(self.DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
                    window.append(res)
                sequences.append(window)
                self.labels.append(self.label_map[action])

        self.sequences = np.array(sequences) # turn the list into a numpy array
        print("SEQUENCE DATA FULLY STRUCTURED")
        

    def build_neural_network_architecture(self):
        """ Builds a deep neural network architecture for the problem that is being solved

        Returns:
            Tensorflow Sequential Model: The model created to solve the problem
        """
        print("NOW BUILDING DEEP NEURAL NETWORK ARCHITECTURE")
        # instantiating the model (the sequential API)
        # it makes easy to make the model by adding the layers
        model = Sequential()
        # 3 sets of lstm layers
        # 64 lstm units, return sequences=True allows the next layer to use the sequence, activation=relu (we can play around with that, we then defined the shape
        model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)))
        model.add(LSTM(128, return_sequences=True, activation='relu'))
        # This is return_sequence=False because the next layers are Dense layers
        # Andrew ings deep learning specialization 
        model.add(LSTM(64, return_sequences=False, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(self.actions.shape[0], activation='softmax')) 

        print("MODEL BUILT")
        return model

ML_Model()