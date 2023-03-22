#!/usr/bin/env python

""" Use this script to convert the annotation of the REAL-Colon dataset from the VOC format to the COCO format.
    For each dataset group (1-4) the first 12 videos will go to the training set, the remaining 3 to the validation set
    Also the code will sample a subset of the negative images to be used for training

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


def convert_video_list(cosmo_base_dataset_folder, video_list, annotation_list, frames_output_folder, json_output_file):
    """
    Takes in input a list of video folders (each of them contains the video frames) and the relative annotation folders and
    convert them into COCO format. All frames with boxes are added to the dataset, while the negative frames are randomly selected
    from the whole dataset. We select N negative frames where N = max(1% of #negative_frames, 10% of #frames_with_boxes)

    Args:
        cosmo_base_dataset_folder (string) : Base folder for the uncompressed REAL-colon dataset in the original format
        video_list (list) : List of video folders to which conversion should be applied
        annotation_list (list) : List of annotation folders to which conversion should be applied
        frames_output_folder (string): Output folders for the frames (relative symlink will be created)
        json_output_file (string): Name of the json output file with the annotation for each frame in the dataset
    """

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
        all_images = sorted(os.listdir(os.path.join(cosmo_base_dataset_folder, curr_video_folder)), key=lambda x : int(x.split("_")[-1].split(".")[0]))
        all_xmls = sorted(os.listdir(os.path.join(cosmo_base_dataset_folder, curr_ann_folder)), key=lambda x : int(x.split("_")[-1].split(".")[0]))
        if not len(all_images) == len(all_xmls):
            raise Exception("Image and annotations must have same length")

        # Only select a subsets of XMLS that are useful for training
        all_datas = []
        num_boxes_indexes = []
        for c_xml in all_xmls:
            c_data = parsevocfile(os.path.join(cosmo_base_dataset_folder, curr_ann_folder, c_xml))
            all_datas.append(c_data)
            num_boxes_indexes.append(len(c_data['boxes']))
        num_frames = len(all_datas)
        frames_wbox_indexes = [idx for idx, v in enumerate(num_boxes_indexes) if v > 0]
        frames_nobox_indexes = [idx for idx, v in enumerate(num_boxes_indexes) if v == 0]

        # Select how many negative frames to keep and randomly sample them
        random.seed(1000)
        to_keep = int(max(0.01 * len(frames_nobox_indexes), min(0.1 * len(frames_wbox_indexes), len(frames_nobox_indexes))))
        selected_frames = frames_wbox_indexes + random.sample(frames_nobox_indexes, to_keep)
        print(f"Sampling {len(selected_frames)} frames from {num_frames} frames {len(frames_nobox_indexes), len(frames_wbox_indexes)} (wbox/nobox)")
        xml_to_be_used = [all_xmls[y] for y in selected_frames]

        # Process each selected frame for current video
        for c_xml in xml_to_be_used:
            c_data = parsevocfile(os.path.join(cosmo_base_dataset_folder, curr_ann_folder, c_xml))
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

            # Create symbolic link
            os.symlink(
                os.path.relpath(os.path.join(cosmo_base_dataset_folder, curr_video_folder,c_data['img_name']),frames_output_folder),
                os.path.join(frames_output_folder, c_data['img_name']))
            # Image process completed, increment id
            image_uniq_id_cnt += 1
    print(f"Processing completed with {image_uniq_id_cnt} images and {image_uniq_box_cnt} boxes")
    with open(json_output_file, 'w') as off:
        json.dump(data, off)

if __name__ == "__main__":

    # read input data
    cosmo_base_dataset_folder = "real_colon_dataset"
    video_list = sorted([x for x in os.listdir(cosmo_base_dataset_folder) if x.endswith("_frames")])
    annotation_list = sorted([x for x in os.listdir(cosmo_base_dataset_folder) if x.endswith("_annotations")])

    # set output folder for coco format annotations
    output_folder = "real_colon_dataset_coco_fmt"
    os.makedirs(output_folder, exist_ok=True)
    train_images_folder = os.path.join(output_folder, "train_images")
    validation_images_folder = os.path.join(output_folder, "validation_images")
    json_output_file_train = os.path.join(output_folder, "train_ann.json")
    json_output_file_validation = os.path.join(output_folder, "validation_ann.json")

    # Perform the conversion:
    # in this example the first 12 videos for each set will go in the train set,
    # the last 3 videos for each set will go in the validation set
    NUM_TRAIN_VIDEOS_PER_SET = 12
    video_list_train = [x for x in video_list if int(x.split("-")[1].split("_")[0]) <= NUM_TRAIN_VIDEOS_PER_SET]
    annotation_list_train = [x for x in annotation_list if int(x.split("-")[1].split("_")[0]) <= NUM_TRAIN_VIDEOS_PER_SET]
    convert_video_list(cosmo_base_dataset_folder, video_list_train, annotation_list_train, train_images_folder,
                       json_output_file_train)
    print("Training subset conversion completed")
    video_list_validation = [x for x in video_list if int(x.split("-")[1].split("_")[0]) > NUM_TRAIN_VIDEOS_PER_SET]
    annotation_list_validation = [x for x in annotation_list if int(x.split("-")[1].split("_")[0]) > NUM_TRAIN_VIDEOS_PER_SET]
    convert_video_list(cosmo_base_dataset_folder, video_list_validation, annotation_list_validation, validation_images_folder,
                       json_output_file_validation)
    print("Validation subset conversion completed")
