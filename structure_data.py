import numpy as np
import os

class Structurer():

    def __init__(self):
        """ Structurer constructor
        """

        # list of actions
        self.actions = np.array(['ayuda', 'clase', 'donde', 'gracias', 'hola', 'necesitar', 'no_entender', 'repetir', 'n-a', 'empty'])
        # action label dictionary
        self.label_map = {label:num for num, label in enumerate(self.actions)}

        # Constants for amount of videos, number of frames and data path
        self.NO_SEQUENCES = 200
        self.SEQUENCE_LENGTH = 30
        self.DATA_PATH = "/Users/codingdan/Documents/University/Semestre 2 2021-2022/Tesina/codigo/ML_model training/MP_Data"

        # Empty arrays for sequence data and label data
        self.sequences, self.labels = [],[]
        # Runs structure_data to join all data
        self.structure_data()


    def structure_data(self):
        """ Goes action by action and adds all collected sequence data to the sequences list
            for each sequence added to sequences, its corresponding label is added to self.labels
        """
        print("NOW STRUCTURING SEQUENCE DATA")
        sequences = []
        # For each action sequence, the data is added to self.sequences and the action label is added to self.labels
        for action in self.actions:
            print(f"Adding {action} data")
            for sequence in range(self.NO_SEQUENCES):
                window = []
                for frame_num in range(self.SEQUENCE_LENGTH):
                    res = np.load(os.path.join(self.DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
                    window.append(res)
                sequences.append(window)
                self.labels.append(self.label_map[action])

        self.sequences = np.array(sequences) # turn the list into a numpy array
        # save both arrays to numpy files
        np.save("/Users/codingdan/Documents/University/Semestre 2 2021-2022/Tesina/codigo/ML_model training/structured_data", self.sequences) 
        np.save("/Users/codingdan/Documents/University/Semestre 2 2021-2022/Tesina/codigo/ML_model training/labels", self.labels)

        print("SEQUENCE DATA FULLY STRUCTURED")

Structurer()