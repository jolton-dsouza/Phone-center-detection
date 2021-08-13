## Phone Detection
Using YOLOv3 to do phone center coordinates detection.

### Getting Started

Use the following instructions to make your system ready to run the code.

### Dependencies

Project is run using:
- Linux Ubuntu >=16.0
- Python >=3.7.1

### Installation

A requirements.txt is added to the 'phone_finder_solution.zip' file which can be used to install the dependencies.

```
pip install -r requirements.txt
```
### Steps
1. Unzip the 'phone_finder_solution.zip' file
2. Run the 'train_phone_finder.py'
3. Run the 'start_training.sh' file
4. Run the 'find_phone.py' file

### Inside the Zip File

The 'phone_finder_solution.zip' file contains python scripts to train and test the model.
It also consists of a 'start_training.sh' file and 'transfer_dataset_to_darknet.py' utility files to start YOLOv3 training.

### Files

- find_phone.py : Python script to test the model. There is an accuracy measurement function added too in the script to find the mean squared error of the predicted vs actual coordinates.
- train_phone_finder.py : Python script to train the model.
- start_training.sh : shell script that will start the YOLOv3 training.
- transfer_dataset_to_darknet.py : Utility python script to transfer the dataset to darknet folder(This need not be separately run, included in 'start_training.sh' script) 
- requirements.txt : Contains list of packages required to run the scripts.
- readme.md
- find_phone_regression : Solving problem as a regression task(not required).

### Folders
- testing_cfg_weights : Contains weight to be used while testing in the 'find_phone.py script'.
- YOLOv3_results : Contains 2 .jpg images, one depicting the YOLOv3 loss and the other is the predicted detection.

### Training Script
- train_phone_finder.py : takes a single command line argument which is a path to the folder with labeled images and labels.txt

### Testing Script
- find_phone.py : takes a single command line argument which is a path to the jpeg image to be tested. Another command line argument of test images folder can be passed to it to use the function 'totalError()' to find mse.
