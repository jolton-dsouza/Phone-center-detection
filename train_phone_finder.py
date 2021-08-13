import cv2
import os
import sys
import shutil

cwd = os.getcwd()

def makeYoloV3Annotation(dataset_path):
    labels_txt = os.path.join(dataset_path + '/labels.txt')
    
    total_imgs = 0
    with open(labels_txt, 'r') as file:
        for line in file:
            total_imgs += 1
    file.close()

    # Divide train set and val set
    if not os.path.exists(os.path.join(cwd, 'train')):
        os.mkdir(os.path.join(cwd, 'train'))

    if not os.path.exists(os.path.join(cwd, 'val')):
        os.mkdir(os.path.join(cwd, 'val'))

    train_percent_div = 0.8

    train_sz = int(total_imgs*train_percent_div)

    # Annotate in yolov3 label format
    count = 0
    dist_from_centre = 20.0/224.0

    with open(labels_txt, 'r') as file:
        for line in file:
            count += 1
            vals = line.split(' ')
            
            img = cv2.imread(os.path.join(dataset_path + '/' + vals[0]))

            if count <= train_sz:
                cv2.imwrite(os.path.join(cwd, 'train/' + "{}.jpg".format(count-1)), img)
                file2 = open(os.path.join(cwd, 'train/' +"{}.txt".format(count-1)),"w")
                
                file2.write("{} {} {} {} {}".format(0, float(vals[1]), float(vals[2]), 2*dist_from_centre, 2*dist_from_centre))

                file2.close()

            else:
                cv2.imwrite(os.path.join(cwd, 'val/' + "{}.jpg".format(count-1)), img)
                file2 = open(os.path.join(cwd, 'val/' +"{}.txt".format(count-1)),"w")
                
                file2.write("{} {} {} {} {}".format(0, float(vals[1]), float(vals[2]), 2*dist_from_centre, 2*dist_from_centre))

                file2.close()

    file.close()
    print('###################### YOLOv3 Train Set and Validation Set created!!! ######################')
    annotated_path = os.path.join(cwd, 'train/')
    zip_file = shutil.make_archive('dataset.zip', 'zip', annotated_path)
    shutil.move(zip_file, annotated_path) 
    

def main():
    # Parse images and labels.txt directory path to make Yolov3 annotation 
    makeYoloV3Annotation(sys.argv[1])

if __name__ == "__main__":
    main()