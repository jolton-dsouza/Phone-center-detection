# Check if system using GPU
nvidia-smi

# Clone Darknet
git clone https://github.com/AlexeyAB/darknet

# Change directory to darknet
cd darknet

# Change and do cmake for GPU config
sed -i 's/OPENCV=0/OPENCV=1/' Makefile
sed -i 's/GPU=0/GPU=1/' Makefile
sed -i 's/CUDNN=0/CUDNN=1/' Makefile
make

# Configure Darknet network for training YOLOv3
cp cfg/yolov3.cfg cfg/yolov3_training.cfg
sed -i 's/batch=1/batch=64/' cfg/yolov3_training.cfg
sed -i 's/subdivisions=1/subdivisions=16/' cfg/yolov3_training.cfg
sed -i 's/max_batches = 500200/max_batches = 4000/' cfg/yolov3_training.cfg
sed -i '610 s@classes=80@classes=1@' cfg/yolov3_training.cfg
sed -i '696 s@classes=80@classes=1@' cfg/yolov3_training.cfg
sed -i '783 s@classes=80@classes=1@' cfg/yolov3_training.cfg
sed -i '603 s@filters=255@filters=18@' cfg/yolov3_training.cfg
sed -i '689 s@filters=255@filters=18@' cfg/yolov3_training.cfg
sed -i '776 s@filters=255@filters=18@' cfg/yolov3_training.cfg

echo "Phone" > data/obj.names
mkdir weights
echo -e 'classes= 1\ntrain  = data/train.txt\nvalid  = data/test.txt\nnames = data/obj.names\nbackup = /weights' > data/obj.data
mkdir data/obj

# Get weights for YOLOv3
wget https://pjreddie.com/media/files/darknet53.conv.74

# Unzip images to data/
unzip ../train/dataset.zip -d data/obj

# Transfer created dataset to darknet folder
cd ..
cp transfer_dataset_to_darknet.py /darknet
cd darknet
./transfer_dataset_to_darknet.py

# Start Training
./darknet detector train data/obj.data cfg/yolov3_training.cfg darknet53.conv.74 -dont_show