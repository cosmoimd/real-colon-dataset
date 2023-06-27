#!/usr/bin/env python

""" Use this script to convert the annotation of the REAL-Colon dataset from the VOC format to the COCO format.
    The script allows to include in the converted dataset a subset of the whole dataset, selecting the number of positive and negative images.
    The script will also produce 3 splits (training, validation, testing), with same proportion across each dataset group (1-4) 

    Copyright 2023-, Cosmo Intelligent Medical Devices
"""

import os
import json
import random
import xml.etree.ElementTree as ET


def parsevocfile(annotation_file):
    """ Parse an annotation file in voc format

        Example VOC notation:
            <annotation>
                </version_fmt>1.0<version_fmt>
                <folder>002-001_frames</folder>
                <filename>002-001_18185.jpg</filename>
                <source>
                    <database>cosmoimd</database>
                    <release>v1.0_20230228</release>
                </source>
                <size>
                    <width>1240</width>
                    <height>1080</height>
                    <depth>3</depth>
                </size>
                <object>
                    <name>lesion</name>
                    <unique_id>videoname_lesionid</unique_id>
                    <box_id>1</box_id>  <- id of the box within the image
                    <bndbox>
                        <xmin>540</xmin>
                        <xmax>1196</xmax>
                        <ymin>852</ymin>
                        <ymax>1070</ymax>
                    </bndbox>
                </object>
            </annotation>""

    Args:
        ann_filename (string) : Full path to the file to parse

    Returns:
        dict: The list of boxes for each class and the image shape
    """

    if not os.path.exists(annotation_file):
        raise Exception("Cannot find bounding box file %s" % (annotation_file))
    try:
        tree = ET.parse(annotation_file)
    except Exception as e:
        print(e)
        raise Exception("Failed to open annotation file %s" % annotation_file)

    # Read all the boxes
    img = {}
    cboxes = []
    for elem in tree.iter():
        #Get the image full path from the image name and folder, not from the annotation tag
        if 'filename' in elem.tag:
            filename = elem.text
        if 'width' in elem.tag:
            img['width'] = int(elem.text)
        if 'height' in elem.tag:
            img['height'] = int(elem.text)
        if 'depth' in elem.tag:
            img['depth'] = int(elem.text)
        if 'object' in elem.tag or 'part' in elem.tag:
            obj = {}

            # create empty dict where store properties
            for attr in list(elem):
                if 'name' in attr.tag:
                    obj['name'] = attr.text
                if 'unique_id' in attr.tag:
                    obj['unique_id'] = attr.text

                if 'bndbox' in attr.tag:
                    for dim in list(attr):
                        if 'xmin' in dim.tag:
                            l = int(round(float(dim.text)))
                        if 'ymin' in dim.tag:
                            t = int(round(float(dim.text)))
                        if 'xmax' in dim.tag:
                            r = int(round(float(dim.text)))
                        if 'ymax' in dim.tag:
                            b = int(round(float(dim.text)))

                    obj["box_ltrb"] = [l, t, r, b]
            cboxes.append(obj)
    img_shape = (img["height"], img["width"], img["depth"])
    return {"boxes": cboxes, "img_shape": img_shape, "img_name": filename }


def convert_video_list(base_dataset_folder, video_list, annotation_list, frames_output_folder, json_output_file, negative_ratio=0, num_positives_per_lesions=-1):
    """
    Takes in input a list of video folders (each of them contains the video frames) and the relative annotation folders and
    convert them into COCO format. All frames with boxes are added to the dataset, while the negative frames are randomly selected
    from the whole dataset. We select N negative frames where N = max(1% of #negative_frames, 10% of #frames_with_boxes)

    Args:
        base_dataset_folder (string) : Base folder for the uncompressed REAL-colon dataset in the original format
        video_list (list) : List of video folders to which conversion should be applied
        annotation_list (list) : List of annotation folders to which conversion should be applied
        frames_output_folder (string): Output folders for the frames (relative symlink will be created)
        json_output_file (string): Name of the json output file with the annotation for each frame in the dataset
        negative_ratio (float): Ratio of frames without boxes to keep for each video (must be in [0,1])
        num_positives_per_lesions (int): how many frames to keep for each lesion (-1 = keep all of them)
    """

    # Check input parameters are valid
    if negative_ratio < 0 or negative_ratio > 1:
        raise Exception(f"Invalid 'negative_ratio' arg {negative_ratio}, must be in [0,1]")

    # Hardcoded dictionary fields
    data = {}
    data['info'] = {'description': 'Cosmo data', 'url': 'http://cosmoimd.com', 'version': '1.0', 'year': 2023, 'contributor': 'CosmoIMD', 'date_created': '2023/02/28'}
    data['licenses'] = [{'url': 'https://creativecommons.org/licenses/by-nc-sa/4.0/', 'id': 1, 'name': 'Attribution-NonCommercial-ShareAlike License'}]
    data['categories'] = [{'supercategory': 'lesion', 'id': 1, 'name': 'lesion'}]
    data['images'] = []
    data['annotations'] = []

    # create output folder
    os.makedirs(frames_output_folder, exist_ok=True)

    # Process each video: subsample frames and convert
    images_uniq_id = {}
    image_uniq_id_cnt = 0
    image_uniq_box_cnt = 0
    uniq_box_to_lesion_association = {}
    for video_idx, (curr_video_folder, curr_ann_folder) in enumerate(zip(video_list, annotation_list)):
        print(f"Processing video {video_idx}")
        all_images = sorted(os.listdir(os.path.join(base_dataset_folder, curr_video_folder)), key=lambda x : int(x.split("_")[-1].split(".")[0]))
        all_xmls = sorted(os.listdir(os.path.join(base_dataset_folder, curr_ann_folder)), key=lambda x : int(x.split("_")[-1].split(".")[0]))
        if not len(all_images) == len(all_xmls):
            raise Exception("Image and annotations must have same length")

        # Only select a subsets of XMLS that are useful for training
        all_datas = []
        num_boxes_indexes = []
        for c_xml in all_xmls:
            c_data = parsevocfile(os.path.join(base_dataset_folder, curr_ann_folder, c_xml))
            all_datas.append(c_data)
            num_boxes_indexes.append(len(c_data['boxes']))
        num_frames = len(all_datas)

        # prepare a dictionary with the list of frames for each lesion
        frames_wbox_indexes = [idx for idx, v in enumerate(num_boxes_indexes) if v > 0]
        frames_nobox_indexes = [idx for idx, v in enumerate(num_boxes_indexes) if v == 0]
        per_lesion_dict = {}
        for cidx, c_data in enumerate(all_datas):
            for cbox in c_data['boxes']:
                cname = cbox['unique_id']
                if not cname in per_lesion_dict.keys():
                    per_lesion_dict[cname] = []
                per_lesion_dict[cname].append(cidx)
        print(f"Found {len(per_lesion_dict)} lesions with {' - '.join([str(len(per_lesion_dict[x])) for x in per_lesion_dict.keys()])} frames each")

        # Select the positive samples
        random.seed(1000)
        selected_frames_w_box_indexes = set([])
        for l in per_lesion_dict.keys():
            c_list = per_lesion_dict[l]
            if num_positives_per_lesions > 0:
                random.shuffle(c_list)
                to_select = min(len(c_list), num_positives_per_lesions)
                selected_frames_w_box_indexes = selected_frames_w_box_indexes.union(set(c_list[:to_select]))
            else:
                selected_frames_w_box_indexes = selected_frames_w_box_indexes.union(set(c_list))
        selected_frames_w_box_indexes = sorted(list(selected_frames_w_box_indexes))
        print(
            f"Sampled {num_positives_per_lesions} positive frames per lesion, using {len(selected_frames_w_box_indexes)}/{len(frames_wbox_indexes)} positive frames")

        # Select the negative samples
        to_keep = int(negative_ratio * len(frames_nobox_indexes))
        selected_frames = selected_frames_w_box_indexes + random.sample(frames_nobox_indexes, to_keep)
        print(f"Sampled {to_keep} negative frames from frames {len(frames_nobox_indexes)} total negatives (negative_ratio = {negative_ratio})")
        xml_to_be_used = [all_xmls[y] for y in selected_frames]

        # Process each selected frame for current video
        for c_xml in xml_to_be_used:
            c_data = parsevocfile(os.path.join(base_dataset_folder, curr_ann_folder, c_xml))
            
            # Add the image to the list of images
            data["images"].append({'license': 1, 'file_name': c_data['img_name'], 'height': c_data['img_shape'][0], 'width': c_data['img_shape'][1], 'id': image_uniq_id_cnt})
            images_uniq_id[image_uniq_id_cnt] = c_data['img_name']
            
            # Loop on boxes
            for cbox in c_data['boxes']:
                l = min(cbox['box_ltrb'][0], c_data['img_shape'][1]-1)
                t = min(cbox['box_ltrb'][1], c_data['img_shape'][0]-1)
                r = min(cbox['box_ltrb'][2], c_data['img_shape'][1]-1)
                b = min(cbox['box_ltrb'][3], c_data['img_shape'][0]-1)
                area = (b-t)*(r-l)
                data['annotations'].append({'segmentation': [[l, t, r, t, r, b, l, b]],'area': area,
                                            'iscrowd': 0, 'image_id': image_uniq_id_cnt,
                                            'bbox': [l, t, r-l, b-t], 'category_id': 1, 'id': image_uniq_box_cnt})
                if not cbox['unique_id'] in uniq_box_to_lesion_association.keys():
                    uniq_box_to_lesion_association[cbox['unique_id']] = []
                uniq_box_to_lesion_association[cbox['unique_id']].append(image_uniq_box_cnt)
                image_uniq_box_cnt += 1

            # Create symbolic link from the original dataset location
            os.symlink(
                os.path.relpath(os.path.join(base_dataset_folder, curr_video_folder,c_data['img_name']),frames_output_folder),
                os.path.join(frames_output_folder, c_data['img_name']))
            # Image process completed, increment id
            image_uniq_id_cnt += 1
    print(f"Processing completed with {image_uniq_id_cnt} images and {image_uniq_box_cnt} boxes")
    with open(json_output_file, 'w') as off:
        json.dump(data, off)

if __name__ == "__main__":


    # Parameters
    base_dataset_folder = "/path/to/dataset/folder"  # Path to the folder of the original REAL-COLON dataset (update with proper value)
    num_positives_per_lesions = 1000 # Number of frames with boxes for each polyp to be included in the output dataset
    negative_ratio = 0 # Ratio of images without boxes for each video to be included in the output dataset [0,1]
    NUM_TRAIN_VIDEOS_PER_SET = 10 # the first 10 videos for each set will go in the train set
    NUM_VALID_VIDEOS_PER_SET = 2 # the next 2 in the validation set, and the remaining videos (3)for each set will go in the test set
    output_folder = f"./real_colon_dataset_coco_fmt_3subsets_poslesion{num_positives_per_lesions}_negratio{negative_ratio}" # Output folder for the converted dataset

    # read input data
    video_list = sorted([x for x in os.listdir(base_dataset_folder) if x.endswith("_frames")])
    annotation_list = sorted([x for x in os.listdir(base_dataset_folder) if x.endswith("_annotations")])

    # set output folder for coco format annotations
    os.makedirs(output_folder, exist_ok=False)
    train_images_folder = os.path.join(output_folder, "train_images")
    validation_images_folder = os.path.join(output_folder, "validation_images")
    test_images_folder = os.path.join(output_folder, "test_images")
    json_output_file_train = os.path.join(output_folder, "train_ann.json")
    json_output_file_validation = os.path.join(output_folder, "validation_ann.json")
    json_output_file_test = os.path.join(output_folder, "test_ann.json")

    # Perform the conversion:
    video_list_train = [x for x in video_list if int(x.split("-")[1].split("_")[0]) <= NUM_TRAIN_VIDEOS_PER_SET]
    annotation_list_train = [x for x in annotation_list if int(x.split("-")[1].split("_")[0]) <= NUM_TRAIN_VIDEOS_PER_SET]
    convert_video_list(base_dataset_folder, video_list_train, annotation_list_train, train_images_folder,
                       json_output_file_train,negative_ratio=negative_ratio, num_positives_per_lesions=num_positives_per_lesions)
    print("Training subset conversion completed")
    video_list_validation = [x for x in video_list if (int(x.split("-")[1].split("_")[0]) > NUM_TRAIN_VIDEOS_PER_SET and int(x.split("-")[1].split("_")[0]) <= (NUM_VALID_VIDEOS_PER_SET+NUM_TRAIN_VIDEOS_PER_SET))]
    annotation_list_validation = [x for x in annotation_list if (int(x.split("-")[1].split("_")[0]) > NUM_TRAIN_VIDEOS_PER_SET and int(x.split("-")[1].split("_")[0]) <= (NUM_VALID_VIDEOS_PER_SET+NUM_TRAIN_VIDEOS_PER_SET))]
    convert_video_list(base_dataset_folder, video_list_validation, annotation_list_validation, validation_images_folder,
                       json_output_file_validation,negative_ratio=negative_ratio, num_positives_per_lesions=num_positives_per_lesions)
    print("Validation subset conversion completed")
    video_list_test = [x for x in video_list if int(x.split("-")[1].split("_")[0]) > (NUM_VALID_VIDEOS_PER_SET+NUM_TRAIN_VIDEOS_PER_SET)]
    annotation_list_test = [x for x in annotation_list if int(x.split("-")[1].split("_")[0]) > (NUM_VALID_VIDEOS_PER_SET+NUM_TRAIN_VIDEOS_PER_SET)]
    convert_video_list(base_dataset_folder, video_list_test, annotation_list_test, test_images_folder,
                       json_output_file_test,negative_ratio=negative_ratio, num_positives_per_lesions=num_positives_per_lesions)
    print("Testing subset conversion completed")
