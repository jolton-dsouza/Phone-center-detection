## Phone Detection
Using YOLOv3 to do phone center coordinates detection. Tried the regression approach to solve the problem too(see 'find_phone_regression.py' file). The results with solving complex image data like this is substandard because it cannot learn the features.

### Getting Started

Use the following instructions to make your system ready to run the code.

### Dependencies

Project is run using:
- Linux Ubuntu >=16.0
- Python >=3.7.1

### Installation

Install the dependencies.

```
pip install -r requirements.txt
```
### Steps

1. Run the 'train_phone_finder.py'
2. Run the 'start_training.sh' file
3. Run the 'find_phone.py' file

### Files

- find_phone.py : Python script to test the model. There is an accuracy measurement function added too in the script to find the mean squared error of the predicted vs actual coordinates.
- train_phone_finder.py : Python script to train the model.
- start_training.sh : shell script that will start the YOLOv3 training.
- transfer_dataset_to_darknet.py : Utility python script to transfer the dataset to darknet folder(This need not be separately run, included in 'start_training.sh' script) 
- requirements.txt : Contains list of packages required to run the scripts.
- readme.md
- find_phone_regression : Solving problem as a regression task(not required).

### Folders

- YOLOv3_results : Contains 2 .jpg images, one depicting the YOLOv3 loss and the other is the predicted detection.

### Training Script
- train_phone_finder.py : takes a single command line argument which is a path to the folder with labeled images and labels.txt

### Testing Script
- find_phone.py : takes a single command line argument which is a path to the jpeg image to be tested. Another command line argument of test images folder can be passed to it to use the function 'totalError()' to find mse.
