# REAL-colon dataset

## Description
The REAL (Real-world multi-center Endoscopy Annotated video Library) - colon dataset
is composed of 60 recordings of real-world colonoscopies. The recordings comes from 4 different
clinical studies (001 to 004), each contributing with 15 videos.
For each patient/video, several clinical variables have been collected, including endoscope_brand, bowel cleanliness score (BBPS), number of surgically removed colon lesions, etc.
Each removed lesion has been annotated with a bounding box in each video frame where it appeared by trained labelers, supervised by expert gastroenterologists. Polyp information including histology, size and anatomical site has been recorded.

For full details on the dataset and to cite this work, please refer to: Carlo Biffi, Giulio Antonelli, Sebastian Bernhofer, Cesare Hassan, Daizen Hirata, Mineo Iwatate, Andreas Maieron, Pietro Salvagnini, and Andrea Cherubini. "REAL-Colon: A dataset for developing real-world AI applications in colonoscopy." arXiv preprint arXiv:2403.02163 (2024). Available at: https://arxiv.org/abs/2403.02163.

Key stats:
- 60 recordings, 15 for each of the 4 centers
- 2757723 total frames
- 132 removed colorectal polyps
- 351264 bounding box annotations

## Data Format
The dataset is composed by the following files:
- 60 compressed folders named `{SSS}-{VVV}_frames` with the frames from each recording
- 60 compressed folders named `{SSS}-{VVV}_annotation` with the annotations from each recordings
- video_info.csv file, a file with the metadata for each video
- lesion_info.csv, a file with the metadata for each lesion
- dataset_description.md, a readme file with information about the dataset


### Annotation file
Each xml file has this format:
Sample VOC notation:
    <annotation>
        </version_fmt>1.0<version_fmt>
        <folder>string</folder>
        <filename>SSS-VVV_t.jpg</filename>
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
    </annotation>

### CSV files
The two csv files contain the metadata for each video (video_info.csv) and for
each lesion (lesion_info.csv).

The `video_info.csv` file reports the following data:
- unique_video_name: name of the video in the format SSS-VVV (site-video)
- age: patient's age
- sex: patient's sex
- endoscope_brand: Endoscope brand (Fuji or Olympus)
- fps: video fps
- num_frames: video length in frames
- num_lesions: number of removed lesions
- bbps: Boston Bowel Preparation Score

The `lesion_info.csv` file reports the following data:
- unique_object_id: lesion id in the format unique_video_name_x
- unique_video_name: name of the video to which the lesion belongs
- size [mm]: lesion size in mm
- site: anatomical site where the removed lesion was found
- histology_extended: lesion histology
- histology_class: lesion histology assigned to one of the following classes: (adenoma, AD; hyperplastic polyp, HP; sessile serrated lesion, SSL, NO POLYP, traditional serrated adenoma, TSA; OTHER)

## Dataset Download
Run `figshare_dataset.py` to automatically download the dataset in full from Figshare to the `./dataset` folder. Output folder can be updated setting variable `DOWNLOAD_DIR` in `figshare_dataset.py`. Dataset files are roughly 1TB worth of information and, depending on your resources, can take up to multiple days to download, so please make proper preparations.

The dataset has also been uploaded to Figshare [https://plus.figshare.com/articles/media/REAL-colon_dataset/22202866](https://plus.figshare.com/articles/media/REAL-colon_dataset/22202866)  and archived at this DOI:  [10.25452/figshare.plus.22202866
](https://doi.org/10.25452/figshare.plus.22202866)

## Dataset Exploration and Stats
To visualize some random exemplar images of the datasets together with clinical information about the data a Jupyter Notebook file, `explore_data.ipynb`, has been prepared.

To reproduce statistics and plots reported in the paper "REAL-Colon: A dataset for developing real-world AI applications in colonoscopy" (https://arxiv.org/abs/2403.02163) please refer to the two python codes: `dataset_stats.py` and `bounding_boxes_plots.py`.

## Polyp Detection
To format data in the COCO format to be used for AI models training and validation and reproduce the results obtained in the paper "REAL-Colon: A dataset for developing real-world AI applications in colonoscopy", please refer to: [Polyp Detection](polyp_detection/README.md)

## Version
v1.0, 2023/02/28
v2.0, 2023/03/01

## Contact
Andrea Cherubini - acherubini@cosmoimd.com

## License
CC BY 4.0, https://creativecommons.org/licenses/by/4.0/

