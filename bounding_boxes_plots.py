#!/usr/bin/env python

""" Compute and save plots that demonstrate data distributions of the bounding box annotations in
    the REAL-Colon dataset.

    Usage:
        - Update base_dataset_path = "/path/to/dataset/folder" with path to the folder containing the REAL-colon dataset
        - python3 bounding_boxes_plots.py

    Copyright 2023-, Cosmo Intelligent Medical Devices
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MaxNLocator
import seaborn as sns
import numpy as np
import concurrent.futures

# Import repo scripts
from polyp_detection import export_coco_format


def get_annotation_data(annotation_folder):
    """
    Process video annotation data from a video folder.

    Args:
        annotation_folder (str): The folder path where annotation xml files of a video are located.

    Returns:
        tuple: A tuple containing the following data structures:
            - box_dict (dict): A dictionary where each key is a unique polyp identifier and each value is a list of
            frame IDs associated with the polyp.
            - frame_list (list): A list containing the number of bounding boxes in each frame of the video.
            - concurrent_polyps_first_second (list): A list containing the number of different bounding box annotations in
              the frames belonging to the first second of a polyp's appearance in the video.
    """
    print(f"Processing annotations in folder {annotation_folder}")

    frame_list = []  # Tracks number of bounding boxes per frame
    box_dict = {}  # Maps unique polyp IDs to their bounding box details and corresponding frame IDs

    # Tracks the number of concurrent bounding boxes in the frames belonging to the first second of a polyp's appearance
    concurrent_polyps_first_second = []

    # Iterate over annotation files in chronological order
    ordered_files = sorted(os.listdir(annotation_folder), key=lambda x: int(x.rsplit('_', 1)[-1].split('.')[0]))

    for file in ordered_files:
        # Parse annotation data from file
        ann_data = export_coco_format.parsevocfile(os.path.join(annotation_folder, file))
        frame_list.append(len(ann_data["boxes"]))

        # Process bounding boxes if present
        if ann_data["boxes"]:
            frame_id = int(ann_data["img_name"].rsplit('_', 1)[-1].split('.')[0])

            # Process each bounding box
            for box in ann_data["boxes"]:
                unique_id = box['unique_id']

                # Initialize list for unique ID if not already present
                if unique_id not in box_dict:
                    box_dict[unique_id] = []

                # Track concurrent bounding boxes for the first 150 frames of a polyp's appearance
                if len(box_dict[unique_id]) < 150:
                    concurrent_polyps_first_second.append(len(ann_data["boxes"]))

                # Store bounding box details and corresponding frame ID
                box_dict[unique_id].append([box['box_ltrb'], frame_id])

    return box_dict, frame_list, concurrent_polyps_first_second


def scale_bounding_box(box, original_resolution, target_resolution=(1920, 1080)):
    """
    Scales the bounding box coordinates from their original resolution to a target resolution.

    This function scales a bounding box from an original image resolution to a target resolution.
    The scaling is done by calculating separate scaling factors for the width and height.

    Args:
        box (tuple): A tuple containing the bounding box coordinates in the format (left, top, right, bottom).
        original_resolution (tuple): A tuple containing the original image's resolution in the format (height, width).
        target_resolution (tuple): A tuple containing the target resolution in the format (height, width). Default is (1920, 1080).

    Returns:
        tuple: A tuple of the scaled bounding box coordinates in the format left_scaled, top_scaled, right_scaled, bottom_scaled.
    """

    # Unpack the bounding box coordinates and resolutions
    left, top, right, bottom = box
    original_height, original_width = original_resolution
    target_height, target_width = target_resolution

    # Compute the scaling factors for width and height
    width_scale = target_width / original_width
    height_scale = target_height / original_height

    # Scale the bounding box coordinates using the calculated scaling factors
    left_scaled = int(left * width_scale)
    top_scaled = int(top * height_scale)
    right_scaled = int(right * width_scale)
    bottom_scaled = int(bottom * height_scale)

    return left_scaled, top_scaled, right_scaled, bottom_scaled


def main():
    # Specify here dataset base path
    base_dataset_path = "/path/to/dataset/folder"
    lesion_info_csv = pd.read_csv(os.path.join(base_dataset_path, "lesion_info.csv"))
    video_info_csv = pd.read_csv(os.path.join(base_dataset_path, "video_info.csv"))

    # Create the stats folder if it doesn't exist
    path_ext = "./stats"
    if not os.path.exists(path_ext):
        os.makedirs(path_ext)

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
            round(video_info_csv[video_info_csv["unique_video_name"] == afolder[-19:-12]]["fps"]))

    # Run concurrently the get_annotation_data over video folders to load video annotation data
    combined_result_dict = {}
    combined_result_list = []
    tot_list_number_of_concurrent_polyps_first_second = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=70) as executor:
        result_concurrent = executor.map(get_annotation_data, annotation_folders)
    for result in result_concurrent:
        result_dict, result_list, list_number_of_concurrent_polyps_first_second = result
        tot_list_number_of_concurrent_polyps_first_second += list_number_of_concurrent_polyps_first_second
        combined_result_list += result_list
        for key, value in result_dict.items():
            if key in combined_result_dict:
                combined_result_dict[key].extend(value)
            else:
                combined_result_dict[key] = value

    # Section: Heatmaps and Box Plots of Polyp BB characteristics

    # Plot heatmaps of polyp spatial appearance
    target_resolution = (1080, 1352)  # rescale all the frame resolutions to this target resolution
    heatmaps = [np.zeros(target_resolution), np.zeros(target_resolution)]
    bins = [0, 1, 2, 3, 4, 5, 6, 7, 8, float('inf')]
    labels = [str(i) for i in range(1, 9)] + ["10+"]
    bbarea_labels = ["t <= 1s", "t > 1s"]  # study two time intervals
    bounding_box_areas_first1 = {}
    bounding_box_areas_over_1 = {}
    for key, boxes in combined_result_dict.items():
        # Retrieve the original resolution for video of name = key
        original_resolution = (resolutions[key[:-2]][0], resolutions[key[:-2]][1])

        # Rescale bounding boxes to target resolution and update the heatmaps
        rescaled_boxes = []
        for i, box in enumerate(boxes):
            left_scaled, top_scaled, right_scaled, bottom_scaled = \
                scale_bounding_box(box[0], original_resolution, target_resolution)
            rescaled_boxes.append([left_scaled, top_scaled, right_scaled, bottom_scaled])

            # t <= 1s or more depends on the video fps
            if i <= fps[key[:-2]]:
                heatmaps[0][top_scaled:bottom_scaled, left_scaled:right_scaled] += 1
            if i > fps[key[:-2]]:
                heatmaps[1][top_scaled:bottom_scaled, left_scaled:right_scaled] += 1

        # Save in a list the bb areas for the video
        boxes = rescaled_boxes
        bounding_box_areas_first1[key] = [
            (box[2] - box[0]) * (box[3] - box[1]) / (target_resolution[0] * target_resolution[1])
            for box in boxes[:fps[key[:-2]]]]
        bounding_box_areas_over_1[key] = [
            (box[2] - box[0]) * (box[3] - box[1]) / (target_resolution[0] * target_resolution[1])
            for box in boxes[fps[key[:-2]]:]]

    # Add a column to the dataframe for the size category
    lesion_info_csv['size_category'] = pd.cut(lesion_info_csv['size [mm]'], bins=bins, labels=labels,
                                              include_lowest=True)

    # Add a column to the dataframe for the bounding box areas
    lesion_info_csv[bbarea_labels[0]] = lesion_info_csv['unique_object_id'].map(
        lambda uid: bounding_box_areas_first1[uid] if uid in bounding_box_areas_first1 else [])
    lesion_info_csv[bbarea_labels[1]] = lesion_info_csv['unique_object_id'].map(
        lambda uid: bounding_box_areas_over_1[uid] if uid in bounding_box_areas_over_1 else [])

    # Create a PDF file to save the boxplots
    pdf_file_path = f"{path_ext}/boxplots.pdf"
    pdf_pages = PdfPages(pdf_file_path)

    # Create a new figure with three subplots arranged vertically
    fig, axs = plt.subplots(2, 1, figsize=(8, 10), sharex=True)

    # Iterate over the bbarea_labels
    for i, bb_text in enumerate(bbarea_labels):
        # Create a new dataframe where each row is a bounding box area
        df_areas = lesion_info_csv.explode(bb_text)
        df_areas[bb_text] = pd.to_numeric(df_areas[bb_text])

        # Create boxplot in the corresponding subplot
        axs[i].boxplot([df_areas.loc[df_areas['size_category'] == label, bb_text].dropna() for label in labels],
                       vert=False,
                       patch_artist=True)
        axs[i].set_yticks(range(1, len(labels) + 1))
        axs[i].set_yticklabels(labels)
        axs[i].set_ylabel('Polyp Size [mm]')
        axs[i].set_title('Polyp Bounding Box Area by Polyp Size for ' + bb_text)

    # Set common x-axis label and tight layout
    plt.xlabel('Polyp Bounding Box Area')
    plt.tight_layout()

    # Save the figure as PDF and close
    pdf_pages.savefig(fig, bbox_inches='tight', dpi=300)
    plt.close()
    pdf_pages.close()

    # Create a PDF file to save the heatmaps
    pdf_file_path = f"{path_ext}/heatmaps.pdf"
    pdf_pages = PdfPages(pdf_file_path)

    fig, axs = plt.subplots(2, 1, figsize=(8, 12))

    # Iterate over the bbarea_labels
    for i, bb_text in enumerate(bbarea_labels):
        # Create a new dataframe where each row is a bounding box area
        df_areas = lesion_info_csv.explode(bb_text)
        df_areas[bb_text] = pd.to_numeric(df_areas[bb_text])

        # Create boxplot in the corresponding subplot
        axs[i].imshow(heatmaps[i], cmap='hot')
        axs[i].set_title('Heatmap of Polyp Spatial Distribution for ' + bb_text)

    # Add the current figure to the PDF
    pdf_pages.savefig(fig, bbox_inches='tight', dpi=300)
    plt.close()

    # Finalize and save the PDF file
    pdf_pages.close()

    # Section: Distribution of Bounding Boxes per Frame

    # Initialize an empty list to hold the number of bounding boxes per frame
    lension_n_boxes = []

    # Populate the list with the number of bounding boxes for each frame
    for key, boxes in combined_result_dict.items():
        lension_n_boxes.append(len(boxes))

    # Create a 1x2 grid of subplots for the two plots
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Adjust the horizontal spacing between plots
    fig.subplots_adjust(wspace=0.3)

    # Section Start: Histogram of Bounding Boxes per Frame

    # Define the histogram bins
    bins = [0, 1, 2, 3, 4, 5]

    # Create a dataframe for the histogram
    s = pd.DataFrame({'val': combined_result_list, 'col': ['1' if x == 0 else '0' for x in combined_result_list]})

    # Plot the histogram of bounding boxes per frame on the first subplot
    hist = sns.histplot(data=s, x="val", hue="col",
                        bins=bins, color='dodgerblue', log_scale=(False, True), edgecolor='darkblue',
                        ax=axes[0], legend=False)

    # Configure the labels and ticks for the first subplot
    axes[0].set_title('Distribution of Bounding Boxes per Frame', fontsize=16)
    axes[0].set_xlabel('Number of Boxes per Frame', fontsize=14)
    axes[0].set_ylabel('Number of Frames (log scale)', fontsize=14)
    axes[0].set_xticks([i + 0.5 for i in range(5)])  # shift x-tick values
    axes[0].set_xticklabels(range(5))  # manually set the x-tick labels

    # Annotate the histogram bars with their respective heights
    for p in hist.patches:
        height = p.get_height()
        axes[0].text(p.get_x() + p.get_width() / 2., height + 0.2, '{:1.0f}'.format(height), ha="center")

    # Section Start: Histogram of Distribution of Bounding Boxes per Polyp

    # Define the logarithmic bin edges
    log_bins = np.logspace(np.log10(100), np.log10(100000), num=25)

    # Plot the histogram of bounding boxes per polyp on the second subplot
    sns.histplot(lension_n_boxes, bins=log_bins, color='lightblue', ax=axes[1])

    # Configure the labels and ticks for the second subplot
    axes[1].set_title("Distribution of Bounding Boxes per Polyp", fontsize=16)
    axes[1].set_xlabel("Number of Boxes per Polyp (log scale)", fontsize=14)
    axes[1].set_ylabel("Number of Polyps", fontsize=14)
    axes[1].set_xscale('log')
    axes[1].set_xlim(10 ** 2, 10 ** 5)

    # Save the figure as a PDF file
    plt.savefig(f"{path_ext}/hist_boxes_per_frame_per_polyp.pdf")

    # Display the figure
    plt.show()

    # Section: Histogram of Broken Tracklets Count

    # Create a 1x2 grid of subplots for the two plots
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Compute the number of broken tracklets for each video
    broken_tracklets = [1 + sum(1 for i, box in enumerate(boxes[:-1]) if boxes[i + 1][1] - box[1] > fps[key[:-2]])
                        for key, boxes in combined_result_dict.items()]

    # Define the histogram bins
    bins = range(min(broken_tracklets), max(broken_tracklets) + 2)  # +2 to include the maximum value

    # Plot the histogram of broken tracklets on the first subplot
    sns.histplot(broken_tracklets, bins=bins, color=sns.color_palette("deep")[2], ax=axes[0])
    axes[0].set_title('Distribution of Polyp Tracklets (Disappearance Threshold = 1 sec)', fontsize=16)
    axes[0].set_xlabel('Number of Tracklets', fontsize=14)
    axes[0].set_ylabel('Number of Polyps', fontsize=14)

    # Configure x and y axes to only have integer ticks
    axes[0].yaxis.set_major_locator(MaxNLocator(integer=True))
    axes[0].xaxis.set_major_locator(MaxNLocator(integer=True))

    # Section: Histogram of Duration of Disappearances

    # Compute the duration of disappearances and the corresponding count for each threshold
    disapperance_length = {}
    for i_l, time in enumerate([l/4 for l in range(1, 56, 1)]):
        disapperance_length[time] = 0
        for key, boxes in combined_result_dict.items():
            for i, box in enumerate(boxes[:-1]):
                # Difference in frame id between two consecutive boxes converted into seconds
                time_difference = (boxes[i + 1][1] - box[1]) / fps[key[:-2]]
                if time <= time_difference < time+0.25:
                    disapperance_length[time] += 1

    # Unpack the thresholds and counts from the data
    thresholds, counts = zip(*disapperance_length.items())

    # Plot the number of tracklets vs disappearance threshold on the second subplot
    sns.lineplot(x=thresholds, y=counts, marker='o', color=sns.color_palette("deep")[3], ax=axes[1])
    axes[1].set_title('Number of Tracklets vs Disappearance Threshold', fontsize=16)
    axes[1].set_xlabel('Disappearance Threshold (seconds)', fontsize=14)
    axes[1].set_ylabel('Number of Tracklets', fontsize=14)

    # Highlight the third data point with a scatter plot
    sns.scatterplot(x=[thresholds[2]], y=[counts[2]], color=sns.color_palette("deep")[2], s=300, ax=axes[1])

    # Configure the plot layout and save the figure as a PDF
    plt.tight_layout()
    plt.savefig(f"{path_ext}/tracklets_plots.pdf")
    plt.show()

    print("Script execution completed.")


if __name__ == '__main__':
    main()
