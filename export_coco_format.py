import os
import json
import explore_data
import random

# We create a dataset in COCO format to be used for model training:
# Approach
#
# Get a fixed number of positive sample per lesion and



def convert_video_list(cosmo_base_dataset_folder, video_list, annotation_list, frames_output_folder, json_output_file):
    data = {}
    data['info'] = {'description': 'Cosmo data', 'url': 'http://cosmoimd.com', 'version': '1.0', 'year': 2023, 'contributor': 'CosmoIMD', 'date_created': '2023/02/28'}
    data['licenses'] = [{'url': 'http://creativecommons.org/licenses/by-nc-sa/2.0/', 'id': 1, 'name': 'Attribution-NonCommercial-ShareAlike License'}]
    data['categories'] = [{'supercategory': 'lesion', 'id': 1, 'name': 'lesion'}]

    #data['images'] = [{'license': 2, 'file_name': '000000015335.jpg', 'coco_url': 'http://images.cocodataset.org/val2017/000000015335.jpg', 'height': 480, 'width': 640, 'date_captured': '2013-11-25 14:00:10', 'flickr_url': 'http://farm6.staticflickr.com/5533/10257288534_c916fafd78_z.jpg', 'id': 15335}]
    #data['annotations'] = [{'segmentation': [[510.66, 423.01, 511.72, 420.03, 510.45, 416.0, 510.34, 413.02]], 'area': 702.1057499999998, 'iscrowd': 0, 'image_id': 289343, 'bbox': [473.07, 395.93, 38.65, 28.67], 'category_id': 18, 'id': 1768}]
    data['images'] = []
    data['annotations'] = []

    os.makedirs(frames_output_folder, exist_ok=True)

    images_uniq_id = {}
    image_uniq_id_cnt = 0
    image_uniq_box_cnt = 0
    uniq_box_to_lesion_association = {} # {"__lesion__ID__": []}
    for video_idx, (curr_video_folder, curr_ann_folder) in enumerate(zip(video_list, annotation_list)):

        if video_idx < 5:
            set_name = "train"
        else:
            set_name = "validation"
        all_images = sorted(os.listdir(os.path.join(cosmo_base_dataset_folder, curr_video_folder)), key=lambda x : int(x.split("_")[-1].split(".")[0]))
        all_xmls = sorted(os.listdir(os.path.join(cosmo_base_dataset_folder, curr_ann_folder)), key=lambda x : int(x.split("_")[-1].split(".")[0]))
        if not len(all_images) == len(all_xmls):
            #raise Exception("Image and annotations must have same length")
            pass

        # Only select a subsets of XMLS that are useful for training
        xml_to_be_used = []
        all_datas = []
        num_boxes_indexes = []
        for c_xml in all_xmls:
            c_data = explore_data.parsevocfile(os.path.join(cosmo_base_dataset_folder, curr_ann_folder, c_xml))
            all_datas.append(c_data)
            num_boxes_indexes.append(len(c_data['boxes']))
        num_frames = len(all_datas)
        num_empty_frames = num_boxes_indexes.count(0)
        frames_wbox_indexes = [idx for idx, v in enumerate(num_boxes_indexes) if v > 0]
        frames_nobox_indexes = [idx for idx, v in enumerate(num_boxes_indexes) if v == 0]
        random.seed(1000)
        to_keep = int(max(0.01 * len(frames_nobox_indexes), min(0.1 * len(frames_wbox_indexes), len(frames_nobox_indexes))))
        selected_frames = frames_wbox_indexes + random.sample(frames_nobox_indexes, to_keep)
        print(f"Sampling {len(selected_frames)} frames from {num_frames} frames {len(frames_nobox_indexes), len(frames_wbox_indexes)} (wbox/nobox)")
        xml_to_be_used = [all_xmls[y] for y in selected_frames]

        for c_xml in xml_to_be_used:
            c_data = explore_data.parsevocfile(os.path.join(cosmo_base_dataset_folder, curr_ann_folder, c_xml))
            print("xml loaded")
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

    cosmo_base_dataset_folder = "/home/pietro.salvagnini/storage/projects/_95_public_dataset/cosmo_colon_v20230216_change_only"
    output_folder = "/home/pietro.salvagnini/storage/projects/_95_public_dataset/cosmo_colon_v20230216_change_only_coco_fmt"
    #cosmo_base_dataset_folder = "/ssd/storage/shared/pietro/projects/_95_public_dataset/cosmo_colon_v20230216_change_only"
    #output_folder = "/ssd/storage/shared/pietro/projects/_95_public_dataset/cosmo_colon_v20230216_change_only_coco_fmt"
    video_list = sorted([x for x in os.listdir(cosmo_base_dataset_folder) if x.endswith("_frames")])
    annotation_list = sorted([x for x in os.listdir(cosmo_base_dataset_folder) if x.endswith("_annotations")])

    # coco output folder

    os.makedirs(output_folder, exist_ok=True)
    train_images_folder = os.path.join(output_folder, "train_images")
    validation_images_folder = os.path.join(output_folder, "validation_images")
    json_output_file_train = os.path.join(output_folder, "train_ann.json")
    json_output_file_validation = os.path.join(output_folder, "validation_ann.json")

    video_list_train = video_list[:2]
    annotation_list_train = annotation_list[:2]
    convert_video_list(cosmo_base_dataset_folder, video_list_train, annotation_list_train, train_images_folder,
                       json_output_file_train)

    video_list_validation = video_list[2:]
    annotation_list_validation = annotation_list[2:]
    convert_video_list(cosmo_base_dataset_folder, video_list_validation, annotation_list_validation, validation_images_folder,
                       json_output_file_validation)
