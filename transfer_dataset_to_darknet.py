import glob
import os
import re

if __name__ == "__main__":
    txt_file_paths = glob.glob(r"data/obj/*.txt")
    for i, file_path in enumerate(txt_file_paths):
        with open(file_path, "r") as f_o:
            lines = f_o.readlines()

            text_converted = []
            for line in lines:
                print(line)
                numbers = re.findall("[0-9.]+", line)
                print(numbers)
                if numbers:
                    # Define coordinates
                    text = "{} {} {} {} {}".format(0, numbers[1], numbers[2], numbers[3], numbers[4])
                    text_converted.append(text)
                    print(i, file_path)
                    print(text)
            # Write file
            with open(file_path, 'w') as fp:
                for item in text_converted:
                    fp.writelines("%s\n" % item)

    images_list = glob.glob("data/obj/*.jpg")
    print(images_list)

    file = open("data/train.txt", "w")
    file.write("\n".join(images_list)) 
    file.close() 

    print('############# YOLOv3 train.txt file created!!! #############')
    print('############# Now starting training!!! #############')
