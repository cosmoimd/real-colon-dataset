#!/usr/bin/env python

"""
    Compute sublist from the test annotations to run experiments on polyp types sublists.
    These sublists include including adenoma versus non-adenoma, diminutive versus non-diminutive,
    and hyperplastic polyps in the sigmoid-rectum, showcasing testing subsets where an optimal AI algorithm should
    demonstrate robust performance.

    Usage:
        - Update base_dataset_path = "/path/to/dataset/folder" with path to the folder containing the REAL-colon dataset
        - python3 filter_annotations.py

    Copyright 2023-, Cosmo Intelligent Medical Devices
"""

import os
import json
import random
import pandas as pd


def read_test_annotations(base_path):
    """
    Read test annotations JSON file from the dataset folder.

    Args:
        base_path (str): The path to the dataset folder.

    Returns:
        dict: Loaded content of the 'test_ann.json' file.
    """
    test_ann_path = os.path.join(base_path, 'test_ann.json')
    with open(test_ann_path, 'r') as file:
        data = json.load(file)
    return data


def categorize_images(dict_annotations):
    """
    Categorize images based on annotations.

    This function categorizes images based on annotations provided in a dictionary.

    Args:
        dict_annotations (dict): A dictionary containing 'images' and 'annotations' keys,
            where 'images' is a list of image data and 'annotations' is a list of annotation data.

    Returns:
        dict: A dictionary mapping unique IDs to lists of image IDs where the unique ID appears.
    """
    # Initialize dictionaries to hold categorized image IDs
    ids_categories = {}

    # Organize annotations by image_id for easy access
    annotations_by_image = {}
    for ann in dict_annotations['annotations']:
        image_id = ann['image_id']
        if image_id not in annotations_by_image:
            annotations_by_image[image_id] = []
        annotations_by_image[image_id].append(ann)

    for image in dict_annotations['images']:
        image_id = image['id']
        if image_id not in annotations_by_image:
            continue  # Skip images without annotations

        # Check bounding box sizes and categorize
        for ann in annotations_by_image[image_id]:
            if ann['unique_id'] in ids_categories:
                ids_categories[ann['unique_id']].append(image_id)
            else:
                ids_categories[ann['unique_id']] = [image_id]

    return ids_categories


def filter_annotations_by_object_id(dict_annotations, list_of_ids):
    """
    Filter annotations based on a list of object IDs.

    This function filters annotations and images in a given annotations dictionary based on a provided list of object IDs.

    Args:
        dict_annotations (dict): A dictionary containing 'images' and 'annotations' keys,
            where 'images' is a list of image data and 'annotations' is a list of annotation data.
        list_of_ids (list): A list of unique IDs to filter the annotations.

    Returns:
        dict: The filtered annotations dictionary with updated 'images' and 'annotations' keys.
    """
    # Step 1: Filter images
    # First, find all image_ids corresponding to unique_ids in the adenoma_list
    image_ids_to_keep = set()

    for annotation in dict_annotations['annotations']:
        if annotation['unique_id'] in list_of_ids:
            image_ids_to_keep.add(annotation['image_id'])

    # Now filter the images to keep only those with ids in image_ids_to_keep
    filtered_images = [image for image in dict_annotations['images'] if image['id'] in image_ids_to_keep]

    # Step 2: Filter annotations
    # Keep only annotations where the image_id is in the list of image_ids_to_keep
    filtered_annotations = [annotation for annotation in dict_annotations['annotations'] if
                            annotation['image_id'] in image_ids_to_keep]

    # Update dict_annotations dictionary to only include the filtered images and annotations
    dict_annotations['images'] = filtered_images
    dict_annotations['annotations'] = filtered_annotations

    return dict_annotations


if __name__ == "__main__":
    # Specify here dataset base path
    base_dataset_folder = "/path/to/dataset/folder"
    clinical_data_csv = pd.read_csv(os.path.join(base_dataset_folder, "lesion_info.csv"))
    video_data_csv = pd.read_csv(os.path.join(base_dataset_folder, "video_info.csv"))
    clinical_data = pd.read_csv(clinical_data_csv)
    video_data = pd.read_csv(video_data_csv)

    # Initialize an empty set to store unique image_ids that have annotations
    annotated_image_ids = set()
    test_annotations = read_test_annotations(base_dataset_folder)
    for annotation in test_annotations['annotations']:
        annotated_image_ids.add(annotation['image_id'])
    filtered_images = [image for image in test_annotations['images'] if image['id'] in annotated_image_ids]
    test_annotations['images'] = filtered_images

    ########################################################
    # Save a testing json with only positive images
    ########################################################
    with open(os.path.join(base_dataset_folder, 'positive_test_ann.json'), 'w') as f:
        json.dump(test_annotations, f)

    ########################################################
    # Save a testing json with only positive images within 1s, 3s and 10s of polyp appearance
    ########################################################
    test_annotations = read_test_annotations(base_dataset_folder)
    for annotation in test_annotations['annotations']:
        annotated_image_ids.add(annotation['image_id'])
    filtered_images = [image for image in test_annotations['images'] if image['id'] in annotated_image_ids]
    test_annotations['images'] = filtered_images
    ids_categories = categorize_images(test_annotations)
    oneslist = []
    for polyp_id in ids_categories.keys():
        image_ids = ids_categories[polyp_id]
        videoname = clinical_data[clinical_data["unique_object_id"] == polyp_id]["unique_video_name"].tolist()[0]
        fps = round(video_data[video_data["unique_video_name"] == videoname]["fps"].tolist()[0])
        oneslist.append(image_ids[:int(1 * fps)])

    oneslist = [item for sublist in oneslist for item in sublist]
    filtered_images_oneslist = [image for image in test_annotations['images'] if image['id'] in oneslist]
    test_annotations['images'] = filtered_images_oneslist
    filtered_annotations = [annotation for annotation in test_annotations['annotations'] if
                            annotation['image_id'] in oneslist]
    test_annotations['annotations'] = filtered_annotations
    print("Saving a test dataset containing only images within first second of polyp appearance. Size is: ",
          len(filtered_images_oneslist))
    with open(os.path.join(base_dataset_folder, '1second_test_ann.json'), 'w') as f:
        json.dump(test_annotations, f)

    test_annotations = read_test_annotations(base_dataset_folder)
    for annotation in test_annotations['annotations']:
        annotated_image_ids.add(annotation['image_id'])
    filtered_images = [image for image in test_annotations['images'] if image['id'] in annotated_image_ids]
    test_annotations['images'] = filtered_images
    ids_categories = categorize_images(test_annotations)
    threeslist = []
    for polyp_id in ids_categories.keys():
        image_ids = ids_categories[polyp_id]
        videoname = clinical_data[clinical_data["unique_object_id"] == polyp_id]["unique_video_name"].tolist()[0]
        fps = round(video_data[video_data["unique_video_name"] == videoname]["fps"].tolist()[0])
        threeslist.append(image_ids[:int(3 * fps)])
    threeslist = [item for sublist in threeslist for item in sublist]
    filtered_images_threeslist= [image for image in test_annotations['images'] if image['id'] in threeslist]
    print("Saving a test dataset containing only images within three second of polyp appearance. Size is: ",
          len(filtered_images_threeslist))
    test_annotations['images'] = filtered_images_threeslist
    filtered_annotations = [annotation for annotation in test_annotations['annotations'] if
                            annotation['image_id'] in threeslist]
    test_annotations['annotations'] = filtered_annotations
    with open(os.path.join(base_dataset_folder, '3seconds_test_ann.json'), 'w') as f:
        json.dump(test_annotations, f)

    ########################################################
    # Save a testing json with 3k random images
    ########################################################
    test_annotations = read_test_annotations(base_dataset_folder)
    for i in range(3):
        test_annotations = read_test_annotations(base_dataset_folder)

        # Randomly sample 10% of the image IDs
        sampled_image_ids = random.sample([image['id'] for image in test_annotations['images']], 3000)

        # Filter images and annotations for the sampled IDs
        test_annotations['images'] = [image for image in test_annotations['images'] if image['id'] in sampled_image_ids]
        test_annotations['annotations'] = [annotation for annotation in test_annotations['annotations'] if
                                           annotation['image_id'] in sampled_image_ids]

        # Keep only positive images
        with open(os.path.join(base_dataset_folder, str(i) + '_sampling_3kpositive_test_ann.json'), 'w') as f:
            json.dump(test_annotations, f)

    ########################################################
    # Save a testing json function of size and histology
    ########################################################
    test_annotations = read_test_annotations(base_dataset_folder)
    ids_categories = categorize_images(test_annotations)
    adenoma_list = []
    nadenoma_list = []
    diminutive_list = []
    non_diminutive_list = []
    others_list = []
    for index, row in clinical_data.iterrows():
        # Check if the unique_object_id is in the keys you're interested in
        if row['unique_object_id'] in ids_categories.keys():
            # Append to adenoma_list if histology_class is AD or TSA
            if row['histology_class'] in ['AD', 'TSA']:
                adenoma_list.append(row['unique_object_id'])
            # Append to nadenoma_list if histology_class is SSL or HP
            elif row['histology_class'] in ['SSL', 'HP']:
                nadenoma_list.append(row['unique_object_id'])
            # Append to others_list for all other histology_class values
            else:
                others_list.append(row['unique_object_id'])

            if float(row["size [mm]"]) < 5.5:
                diminutive_list.append(row['unique_object_id'])
            else:
                non_diminutive_list.append(row['unique_object_id'])

    test_annotations_adenoma = filter_annotations_by_object_id(test_annotations, adenoma_list)
    with open(os.path.join(base_dataset_folder, 'adenoma_test_ann.json'), 'w') as f:
        json.dump(test_annotations_adenoma, f)

    test_annotations = read_test_annotations(base_dataset_folder)
    test_annotations_nadenoma = filter_annotations_by_object_id(test_annotations, nadenoma_list)
    with open(os.path.join(base_dataset_folder, 'nadenoma_test_ann.json'), 'w') as f:
        json.dump(test_annotations_nadenoma, f)

    test_annotations = read_test_annotations(base_dataset_folder)
    test_annotations_dim = filter_annotations_by_object_id(test_annotations, diminutive_list)
    with open(os.path.join(base_dataset_folder, 'diminutive_test_ann.json'), 'w') as f:
        json.dump(test_annotations_dim, f)

    test_annotations = read_test_annotations(base_dataset_folder)
    test_annotations_ndim = filter_annotations_by_object_id(test_annotations, non_diminutive_list)
    with open(os.path.join(base_dataset_folder, 'non_diminutive_test_ann.json'), 'w') as f:
        json.dump(test_annotations_ndim, f)

    ########################################################
    # HISTOLOGY+LOCATION POLYPS
    ########################################################
    test_annotations = read_test_annotations(base_dataset_folder)
    ids_categories = categorize_images(test_annotations)
    hp_sigmarectum_list = []
    adenoma_distal_list = []
    others_polyps_list = []
    for index, row in clinical_data.iterrows():
        # Check if the unique_object_id is in the keys you're interested in
        if row['unique_object_id'] in ids_categories.keys():
            # Append to adenoma_list if histology_class is AD or TSA
            if row['histology_class'] in ['HP'] and row['site'] in ['sigma', 'rectum']:
                hp_sigmarectum_list.append(row['unique_object_id'])
            else:
                others_polyps_list.append(row['unique_object_id'])

    test_annotations = read_test_annotations(base_dataset_folder)
    test_annotations_nadenoma = filter_annotations_by_object_id(test_annotations, hp_sigmarectum_list)
    with open(os.path.join(base_dataset_folder, 'hp_sigmarectum_list.json'), 'w') as f:
        json.dump(test_annotations_nadenoma, f)

    test_annotations = read_test_annotations(base_dataset_folder)
    test_annotations_dim = filter_annotations_by_object_id(test_annotations, others_polyps_list)
    with open(os.path.join(base_dataset_folder, 'others_polyps_list.json'), 'w') as f:
        json.dump(test_annotations_dim, f)