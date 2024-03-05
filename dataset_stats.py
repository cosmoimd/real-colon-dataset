#!/usr/bin/env python

""" Compute and save statistics that demonstrate various characteristics of the frames and bounding box annotations in
    the REAL-Colon dataset.

    Usage:
        - Update base_dataset_path = "/path/to/dataset/folder" with path to the folder containing the REAL-colon dataset
        - python3 dataset_stats.py

    Copyright 2023-, Cosmo Intelligent Medical Devices
"""
import os
import pandas as pd
import concurrent.futures

# Import repo scripts
from polyp_detection import export_coco_format


def get_annotation_data(annotation_folder):
    """
    Process video annotation data from a video folder.

    Args:
        annotation_folder (str): The folder path where annotation xml files of a video are located.

    Returns:
        box_in_frame_dict: A dictionary containing the following information:
            - box_in_frame_dict: A dictionary containing a value of a list of unique binding boxes for each frame
            - box_in_frame_dict[i]: The key i refers to the frame number
            """
    print(f"Processing annotations in folder {annotation_folder}")

    box_in_frame_dict = {}  # Maps each frame with a list of bounding box ids in the frame

    # Iterate over annotation files in chronological order
    ordered_files = sorted(os.listdir(annotation_folder), key=lambda x: int(x.rsplit('_', 1)[-1].split('.')[0]))

    for file in ordered_files:
        # Parse annotation data from file
        ann_data = export_coco_format.parsevocfile(os.path.join(annotation_folder, file))

        # Retrieve the frame id from the image name
        frame_id = ann_data["img_name"].split('.')[0]

        # Initialize the frame in the dictionary with an empty list
        box_in_frame_dict[frame_id] = []

        # Process bounding boxes if present
        if ann_data["boxes"]:

            # Process each bounding box
            for box in ann_data["boxes"]:
                unique_id = box['unique_id']

                # Store bounding box details and corresponding frame ID
                box_in_frame_dict[frame_id].append(unique_id)

    return box_in_frame_dict


def create_frame_tables(class_ann_data, ext_ann_data, lesion_info_csv, ext, is_frames):
    """
        Create the csv tables from the data.

        Args:
            class_ann_data (dict): A dictionary of frames with the different histology types.
            ext_ann_data (dict): A dictionary of frames with the different extended histology classes
            lesion_info_csv (DataFrame): The csv containing info on the lesion data
            ext (str): the extension to add to the front of the csv file name
            is_frames (bool): indicates whether a frames csv or a bounding box csv is to be made
        Generates:
            CSV files listing characteristics of the 4 studies separately and collectively
    """

    # Start create csv files with the total number of frames containing histology classor bounding box types

    # Get unique values from the 'histology_class' column
    unique_histology_classes = lesion_info_csv[['histology_class']].drop_duplicates()

    # Create DataFrames for 'MULTIPLE' 'NEGATIVE FRAME' and 'TOTAL' if frames. Otherwise, just 'TOTAL'
    new_rows = []
    if is_frames:
        new_rows = [{'histology_class': 'MULTIPLE'}, {'histology_class': 'NEGATIVE FRAME'}, {'histology_class': 'TOTAL'}]
    else:
        new_rows = [{'histology_class': 'TOTAL'}]
    new_df = pd.DataFrame(new_rows)

    # Concatenate the new DataFrame with the existing unique_histology_classes DataFrame
    unique_histology_classes = pd.concat([unique_histology_classes, new_df], ignore_index=True)

    # Count the occurrences of each unique 'histology_class' in class_ann_data
    histology_counts = pd.Series(class_ann_data).value_counts()

    # Merge the counts into the unique_histology_classes DataFrame
    new_column_name = ""
    if is_frames:
        new_column_name = 'frames_count'
    else:
        new_column_name = 'bounding_box_count'
    unique_histology_classes = unique_histology_classes.merge(
        histology_counts.rename(new_column_name).reset_index(),
        how='left',
        left_on='histology_class',
        right_on='index'
    )

    # Fill NaN values in 'frames_count' or 'bounding_box_count' with 0
    unique_histology_classes[new_column_name] = unique_histology_classes[new_column_name].fillna(0)

    # Calculate the count for 'TOTAL' by summing up all other count values
    total_count = unique_histology_classes[new_column_name].sum()
    unique_histology_classes.loc[
        unique_histology_classes['histology_class'] == 'TOTAL', new_column_name] = total_count

    # Calculate the 'percentage' column and round
    unique_histology_classes['percentage'] = (unique_histology_classes[new_column_name] / total_count) * 100
    unique_histology_classes['percentage'] = unique_histology_classes['percentage'].round(2)

    # Convert the count column to integers
    unique_histology_classes[new_column_name] = unique_histology_classes[new_column_name].astype(int)

    # Drop the 'index' column
    unique_histology_classes = unique_histology_classes.drop(columns=['index'])

    # Save the DataFrame to a CSV file with the counts
    if is_frames:
        unique_histology_classes.to_csv(f"./stats/{ext}frames_histology_class.csv", index=False)
    else:
        unique_histology_classes.to_csv(f"./stats/{ext}boxes_histology_class.csv", index=False)

    # Start create csv files with the total number of frames containing histology extended or bounding box types

    # Get unique values from the 'histology_class' column
    unique_histology_extended = lesion_info_csv[['histology_extended']].drop_duplicates()

    # Create DataFrames for 'MULTIPLE' 'NEGATIVE FRAME' and 'TOTAL' if frames. Otherwise, just 'TOTAL'
    new_rows = []
    if is_frames:
        new_rows = [{'histology_extended': 'MULTIPLE'}, {'histology_extended': 'NEGATIVE FRAME'},
                    {'histology_extended': 'TOTAL'}]
    else:
        new_rows = [{'histology_extended': 'TOTAL'}]
    new_df = pd.DataFrame(new_rows)

    # Concatenate the new DataFrame with the existing unique_histology_classes DataFrame
    unique_histology_extended = pd.concat([unique_histology_extended, new_df], ignore_index=True)

    # Count the occurrences of each unique 'histology_extended' in ext_ann_data
    histology_counts = pd.Series(ext_ann_data).value_counts()

    # Merge the counts into the unique_histology_classes DataFrame
    new_column_name = ""
    if is_frames:
        new_column_name = 'frames_count'
    else:
        new_column_name = 'bounding_box_count'
    unique_histology_extended = unique_histology_extended.merge(
        histology_counts.rename(new_column_name).reset_index(),
        how='left',
        left_on='histology_extended',
        right_on='index'
    )

    # Fill NaN values in 'frames_count' or 'bounding_box_count' with 0
    unique_histology_extended[new_column_name] = unique_histology_extended[new_column_name].fillna(0)

    # Calculate the frames_count for 'TOTAL' by summing up all other count values
    total_frames_count = unique_histology_extended[new_column_name].sum()
    unique_histology_extended.loc[
        unique_histology_extended['histology_extended'] == 'TOTAL', new_column_name] = total_frames_count
    unique_histology_extended[new_column_name] = unique_histology_extended[new_column_name].astype(int)

    # Calculate the 'percentage' column
    unique_histology_extended['percentage'] = (unique_histology_extended[new_column_name] / total_frames_count) * 100
    unique_histology_extended['percentage'] = unique_histology_extended['percentage'].round(2)

    # Drop the 'index' column if you don't need it
    unique_histology_extended = unique_histology_extended.drop(columns=['index'])

    # Save the DataFrame to a CSV file with the counts
    if is_frames:
        unique_histology_extended.to_csv(f"./stats/{ext}frames_histology_extended.csv", index=False)
    else:
        unique_histology_extended.to_csv(f"./stats/{ext}boxes_histology_extended.csv", index=False)

    return


def main():
    # Specify here dataset base path
    base_dataset_path = "/path/to/dataset/folder"
    lesion_info_csv = pd.read_csv(os.path.join(base_dataset_path, "lesion_info.csv"))
    video_info_csv = pd.read_csv(os.path.join(base_dataset_path, "video_info.csv"))

    # Retrieve annotations folder
    annotation_folders = []
    for dataset in range(1, 5):
        for vv in range(1, 16):
            annotation_folders += [os.path.join(base_dataset_path, f"{dataset:03d}-{vv:03d}" + "_annotations")]

    # Loop over dataset videos to get their resolution and their fps
    resolutions = {}
    fps = {}
    for afolder in annotation_folders:
        c_ann_data = export_coco_format.parsevocfile(os.path.join(afolder, os.listdir(afolder)[0]))
        resolutions[afolder[-19:-12]] = c_ann_data['img_shape']
        fps[afolder[-19:-12]] = int(
            round(video_info_csv[video_info_csv["unique_video_name"] == afolder[-19:-12]]["fps"].iloc[0]))

    # Run concurrently the get_annotation_data over video folders to load video annotation data
    class_ann_data = {}  # variable to hold annotation info from all the videos for the class csv
    ext_ann_data = {}  # variable to hold annotation info from all the videos for the extended class csv
    with concurrent.futures.ProcessPoolExecutor(max_workers=70) as executor:
        result_concurrent = executor.map(get_annotation_data, annotation_folders)

    for result in result_concurrent:
        result_dict = result

        # Loop to populate class_ann_data
        for key, value in result_dict.items():

            # Add the histology types to the list
            if len(value) == 0:
                class_ann_data[key] = 'NEGATIVE FRAME'
            elif len(value) == 1:
                filtered_row = lesion_info_csv.loc[lesion_info_csv['unique_object_id'] == value[0], 'histology_class']

                # Append the result to the class_ann_data list
                class_ann_data[key] = filtered_row.values[0] if not filtered_row.empty else None

            # If more than 1 box, check if they are all the same type and put them in the multiple section if false
            else:

                # Create a variable to check the other types against
                histology_checker = lesion_info_csv.loc[
                    lesion_info_csv['unique_object_id'] == value[0], 'histology_class'].values[0]
                check_passed = True

                # Loop through the other boxes to check if they are all the same types
                for i in range(1, len(value)):
                    class_to_check = lesion_info_csv.loc[
                        lesion_info_csv['unique_object_id'] == value[i], 'histology_class'].values[0]
                    if class_to_check != histology_checker:
                        class_ann_data[key] = 'MULTIPLE'
                        check_passed = False
                        break

                # If they are all the same kind, add it to the data
                if check_passed:
                    class_ann_data[key] = histology_checker

        # Loop to populate ext_ann_data
        for key, value in result_dict.items():

            # Add the histology types to the list
            if len(value) == 0:
                ext_ann_data[key] = 'NEGATIVE FRAME'
            elif len(value) == 1:
                filtered_row = lesion_info_csv.loc[
                    lesion_info_csv['unique_object_id'] == value[0], 'histology_extended']

                # Append the result to the class_ann_data list
                ext_ann_data[key] = filtered_row.values[0] if not filtered_row.empty else None

            # If more than 1 box, check if they are all the same type and put them in the multiple section if false
            else:

                # Create a variable to check the other types against
                histology_checker = lesion_info_csv.loc[
                    lesion_info_csv['unique_object_id'] == value[0], 'histology_extended'].values[0]
                check_passed = True

                # Loop through the other boxes to check if they are all the same types
                for i in range(1, len(value)):
                    class_to_check = lesion_info_csv.loc[
                        lesion_info_csv['unique_object_id'] == value[i], 'histology_extended'].values[0]
                    if class_to_check != histology_checker:
                        ext_ann_data[key] = 'MULTIPLE'
                        check_passed = False
                        break

                # If they are all the same kind, add it to the data
                if check_passed:
                    ext_ann_data[key] = histology_checker

    # Create the stats folder if it doesn't exist
    if not os.path.exists("./stats"):
        os.makedirs("./stats")

    # Create frame csv for the four studies
    for i in range(1, 5):
        print(f"Creating the frames csv for the study 00{i}")
        filtered_class_data = {key: value for key, value in class_ann_data.items() if key.startswith(f"00{i}")}
        filtered_ext_data = {key: value for key, value in ext_ann_data.items() if key.startswith(f"00{i}")}
        create_frame_tables(filtered_class_data, filtered_ext_data, lesion_info_csv, f"00{i}_", True)

    # Create frame csv for a collective of the studies
    print(f"Creating the frames csv for the collective studies")
    create_frame_tables(class_ann_data, ext_ann_data, lesion_info_csv, "", True)

    # Start creating csv files for bounding box histology distribution

    # Initialize the dictionaries to contain key value pairs of the unique box id and the histology types
    class_box_data = lesion_info_csv.set_index('unique_object_id')['histology_class'].to_dict()
    ext_box_data = lesion_info_csv.set_index('unique_object_id')['histology_extended'].to_dict()

    # Create the csv files for the box histology classes
    for i in range(1, 5):
        print(f"Creating the bounding box csv for the study 00{i}")
        filtered_class_data = {key: value for key, value in class_box_data.items() if key.startswith(f"00{i}")}
        filtered_ext_data = {key: value for key, value in ext_box_data.items() if key.startswith(f"00{i}")}
        create_frame_tables(filtered_class_data, filtered_ext_data, lesion_info_csv, f"00{i}_", False)

    # Create bounding box csv for a collective of the studies
    print(f"Creating the bounding box csv for the collective studies")
    create_frame_tables(class_box_data, ext_box_data, lesion_info_csv, "", False)

    print("Script execution completed.")


if __name__ == '__main__':
    main()
