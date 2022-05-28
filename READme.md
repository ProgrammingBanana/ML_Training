# Machine Learning Sing Language Model Training
This project focuses on the development of the Machine Learning model that will interpret sign language

## Files:
* training.py: has the model creation, training, and testing logic
* structure_data.py: Contains the logic for preparing the data and labels for the model


## Installation and Set up
Having an updated version of Python installed on the computer is necessary for the project to work
1. Run the command ```pip install pipenv``` in your command line terminal
2. Download code from the repository
3. Run the command ```pipenv shell``` to start the virtual environment
4. Go to the project location in the command line and run the command ```pipenv install --ignore-pipfile``` to install dependencies named in the pipfile.lock document
5. Add recorded sign data from video capture project to this project
6. Run the command ```python structure_data.py``` to prepare training data and labels

## Running the application
You are now ready to run the application.  To do so, run the command ```python training.py```.  Doing so will open begin training the model and when it finishes training, it will run some tests
