from sklearn.model_selection import train_test_split # Allows us to train on one partition, and then test it on another one
from tensorflow.keras.utils import to_categorical 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
import numpy as np
import os

class ML_Model():
    def __init__(self):
        """ Constructor for ML_Model class. Opens structured data files and sets up the Machine Learning Model
        """

        # List of all the actions and situations the model should interpret
        self.actions = np.array(['ayuda', 'clase', 'donde', 'gracias', 'hola', 'necesitar', 'no_entender', 'repetir', 'n-a', 'empty'])
        
        # Opens structured_data.npy which contains all sequence keypoint data and stores it self.sequences
        self.sequences = np.load("/Users/codingdan/Documents/University/Semestre 2 2021-2022/Tesina/codigo/ML_model training/structured_data.npy")
        # Opens labels.npy which contains all sequence label data and stores it in self.labels
        self.labels = np.load("/Users/codingdan/Documents/University/Semestre 2 2021-2022/Tesina/codigo/ML_model training/labels.npy")

        # Turn the labels into a categorization list. This turns the labels list into a list of list, where each sub-list contains a set of numbers indicating the category
        self.labels_categorized = to_categorical(self.labels).astype(int) 

        # This splits the sequence and label data into two groups, a training group (which will contain most of the data) and a test group that will consist of 5% of the data
        self.seqs_train, self.seqs_test, self.labels_train, self.labels_test = train_test_split(self.sequences, self.labels_categorized, test_size=0.05)

        
        # Creates a log directory to set up tensorboard callbacks to monitor the training
        self.log_dir = os.path.join('Logs')
        #TensorBoard is a web app that comes with tensorflow that allows you to monitor the neural network training
        # Monitor training and accuracy as it's being trained
        self.tb_callback = TensorBoard(log_dir=self.log_dir)

        self.model = self.build_neural_network_architecture()
        self.train()

        self.test()
        

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
        model.add(LSTM(256, return_sequences=True, activation='relu'))
        model.add(LSTM(128, return_sequences=True, activation='relu'))
        # This is return_sequence=False because the next layers are Dense layers
        # Andrew ings deep learning specialization 
        model.add(LSTM(64, return_sequences=False, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(self.actions.shape[0], activation='softmax')) 

        print("MODEL BUILT")
        return model

    def train(self):
        """ Compiles and trains the model
        """

        # We can play around with the optimizer, 
        # loss has to be categorical_crossentropy because we are working with a multiclass classification model
        # the metric is optional but it allows us to track our accuracy as we train
        # optimizer=Adam(learning_rate=0.0001), allows to update the learning rate
        print("COMPILING THE MODEL")
        self.model.compile(
            optimizer='Adamax',
            loss='categorical_crossentropy', 
            metrics=['categorical_accuracy'])
        print("TRAINING THE MODEL")
        print("In the Logs/train folder, use the command: tensorboard --logdir=.")
        self.model.fit(self.seqs_train, self.labels_train, epochs=160, callbacks=[self.tb_callback])
        self.model.save('./actions_models/actions_3.h5')

    def test(self):
        rest = self.model.predict(self.seqs_train)
        print('The value predicted for the first action in the test partition is: {}'.format(self.actions[np.argmax(rest[2])]))
        print('The actual value for the first action in the test partition is: {}'.format(self.actions[np.argmax(self.labels_test[2])]))

        yhat = self.model.predict(self.seqs_test)
        ytrue = np.argmax(self.labels_test, axis=1).tolist()
        yhat  = np.argmax(yhat, axis=1).tolist()
        matrix = multilabel_confusion_matrix(ytrue, yhat)

        print(matrix)
        print(accuracy_score(ytrue, yhat))

ML_Model()