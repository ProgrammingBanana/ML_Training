import numpy as np
import os

class Structurer():

    def __init__(self):
        self.actions = np.array(['ayuda', 'clase', 'donde', 'gracias', 'hola', 'necesitar', 'no_entender', 'repetir', 'n-a', 'empty'])
        self.label_map = {label:num for num, label in enumerate(self.actions)}

        self.no_sequences = 200
        self.sequence_length = 30
        self.DATA_PATH = "/Users/codingdan/Documents/University/Semestre 2 2021-2022/Tesina/codigo/ML_model training/MP_Data"

        self.sequences, self.labels = [],[]
        self.structure_data()


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
        np.save("/Users/codingdan/Documents/University/Semestre 2 2021-2022/Tesina/codigo/ML_model training/structured_data", self.sequences) 
        np.save("/Users/codingdan/Documents/University/Semestre 2 2021-2022/Tesina/codigo/ML_model training/labels", self.labels)

        print("SEQUENCE DATA FULLY STRUCTURED")

Structurer()