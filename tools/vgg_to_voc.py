import pandas as p
import sys
import os
import pprint
import json
import argparse
import xmltodict
import random
from PIL import Image

def mkdir_if_not_exists(dir_path):
    if not os.path.exists(dir_path):
        print("Creating {}".format(dir_path))
        os.makedirs(dir_path)

def check_for_img_dir(dir_path):
    if not os.path.exists(dir_path + "/JPEGImages"):
        permission = 'n'
        if os.path.exists(dir_path + "/imgs"):
            permission = input(
                "\nPermission to rename {0}/imgs/ to {0}/JPEGImages/ (y/n):".format(dir_path))
        if permission.lower() == 'y':
            print("Renaming {0}/imgs/ to {0}/JPEGImages/\n".format(dir_path))
            os.rename(dir_path + "/imgs", dir_path + "/JPEGImages")
        else:
            print("Cannot locate {0}/JPEGImages/".format(dir_path))
            exit(0)

def vgg_dict_to_voc_dict(obj,size):
    bbox = json.loads(obj["region_shape_attributes"])
    xmin = bbox["x"]
    ymin = bbox["y"]
    xmax = xmin + bbox["width"]
    ymax = ymin + bbox["height"]
    if xmin <= 0:
        xmin = 1
    if xmax >= size["width"]:
        xmax = size["width"]-1
    if ymin <= 0:
        ymin = 1
    if ymax >= size["height"]:
        ymax = size["height"]-1
    bndbox = {
    "xmin" : xmin,
    "ymin" : ymin,
    "xmax" : xmax,
    "ymax" : ymax
    }
    if xmin >= size["width"] or ymin >= size["height"]: 
        #print("{}\n{}\n{}\n\n".format(size,bbox,bndbox))
        return False

    name = json.loads(obj["region_attributes"])["class"]
    
    return {
    "name" : name,
    "pose" : "Unspecified",
    "truncated" : 0,
    "difficult" : 0,
    "bndbox" : bndbox
    }

def determine_image_set(images, class_name):
    image_set = {}
    for i in images:
        img = i.split('.')[0]
        objects = [obj["name"] for obj in images[i]["object"]]
        if class_name in objects:
            image_set[img] = 1
        else:
            image_set[img] = -1
    return image_set

def create_image_sets(images, classes):
    image_sets = {}
    for c in classes:
        image_sets[c] = determine_image_set(images, c)
    return image_sets

def write_image_sets(dir_path, image_sets):
    for split in image_sets:
        split_data = image_sets[split]
        for class_name in split_data:
            filename = dir_path + "/{}_{}.txt".format(class_name,split)
            with open(filename,"w+") as f:
                for img in split_data[class_name]:
                    f.write("{} {}\n".format(img, split_data[class_name][img]))
            #print(filename)

def annotation_to_xml(annotation,dir_path):
    filepath = dir_path + annotation["filename"].split('.')[0] + ".xml"
    # Nest dictionary for proper xml output
    data = {"annotation":annotation}
    with open(filepath, "w+") as f:
        f.write(xmltodict.unparse(data, full_document=False, pretty=True))
    #print(filepath)

def sample_without_replace(data, keys):
    sample = {}
    for key in keys:
        sample[key] = data[key]
        del data[key]
    return sample

def split_train(data, train_ratio, val_ratio, test_ratio):
    total = len(data)
    total_train = int(train_ratio*total)
    total_val = int(val_ratio*total)
    # If no test sample is to be created,
    # ensure training and validation cover all samples
    total_test = 0
    if test_ratio == 0.0:
        total_val = total - total_train
    else:
        total_test = total - total_train - total_val

    train = {}
    val = {}
    test = {}

    print("Total samples: {}\
        Training: {}\
        Validation: {}\
        Testing: {}".format(total, total_train,total_val,total_test))

    # Sample without replacement
    train_keys = random.sample(data.keys(), total_train)
    train = sample_without_replace(data, train_keys)
    val_keys = random.sample(data.keys(), total_val)
    val = sample_without_replace(data, val_keys)
    test_keys = random.sample(data.keys(), total_test)
    test = sample_without_replace(data, test_keys)
    return train,val,test

def create_label_map(dir_path, classes):
    filename = dir_path[dir_path.rfind('/')+1:] + "_label_map.pbtxt"
    filepath = dir_path + '/' + filename
    with open(filepath,"w+") as f:
        for i,c in enumerate(classes):
            f.write('''item {{\n  id : {}\n  name : '{}'\n}}\n\n'''.format(i+1,c))

def vgg_to_voc(csv_df, dir_path, classes):
    
    folder = dir_path[dir_path.rfind('/')+1:] 

    # Create a VOC formatted xml for each image in csv
    annotations = {}
    for i,obj in csv_df.iterrows():
        # Check if this is first object processed in this image
        img_file = obj["#filename"]
        if img_file not in annotations:
            img_path = dir_path + "/JPEGImages/" + img_file
            # Skip if image is missing
            try:
                img = Image.open(img_path)
            except:
                print("Cannot open " + img_path)
                continue
            size = img.size
            annotations[img_file] = {
            "folder" : folder,
            "filename" : img_file,
            "size" : {
                "width" : size[0],
                "height" : size[1]
                },
            "object" : []}
        # Parse object attributes
        if obj["region_count"] > 0:
            obj_attr = vgg_dict_to_voc_dict(obj,annotations[img_file]["size"])
            if obj_attr == False:
                continue
            annotations[img_file]["object"].append(obj_attr)
            classes.add(obj_attr["name"])
    return annotations

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", nargs='?', default=0.8, type=float)
    parser.add_argument("--val", nargs='?', default=0.2, type=float)
    parser.add_argument("--test", nargs='?', default=0.0, type=float)
    parser.add_argument("--file", required=True)
    args = parser.parse_args()

    # Filepaths
    csv_filepath = args.file
    dir_path = csv_filepath[:csv_filepath.rfind('/')]
    check_for_img_dir(dir_path)

    # Collect classes while processing csv
    classes = set()

    # Convert VGG generated CSV annotations to
    # Pascal VOC formatted XML annotations
    csv_df = p.read_csv(csv_filepath)
    annotations = vgg_to_voc(csv_df, dir_path, classes)

    # Create output directories
    annot_dir = dir_path + "/Annotations/"
    image_sets_dir = dir_path + "/ImageSets/Main"
    mkdir_if_not_exists(annot_dir)
    mkdir_if_not_exists(image_sets_dir)
    for a in annotations.values():
        annotation_to_xml(a, annot_dir)

    # Split data and create image set text files for each split
    train,val,test = split_train(annotations, args.train, args.val, args.test)
    split_data = {"train":train, "val":val, "test":test}
    image_sets = {}
    for split in split_data:
        image_sets[split] = create_image_sets(split_data[split], classes)
    write_image_sets(image_sets_dir, image_sets)

    create_label_map(dir_path,classes)
