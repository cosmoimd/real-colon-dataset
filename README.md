# REAL-colon dataset

## Description
The REAL (Real-world multi-center Endoscopy Annotated video Library) - colon dataset
is composed of 60 recordings of real-world colonoscopies. The recordings comes from 4 different
clinical studies (001 to 004), each contributing with 15 videos.
For each patient/video, several clinical variables have been collected, including endoscope_brand, bowel cleanliness score (BBPS), number of surgically removed colon lesions, etc.
Each removed lesion has been annotated with a bounding box in each video frame where it appeared by trained labelers, supervised by expert gastroenterologists. Polyp information including histology, size and anatomical site has been recorded.

Key stats:
- 60 recordings, 15 for each of the 4 centers
- 2757723 total frames
- 132 removed colorectal polyps
- 351264 bounding box annotations


## Download
The dataset has been uploaded to Figshare and archived at this DOI:  [10.25452/figshare.plus.22202866
](https://doi.org/10.25452/figshare.plus.22202866)

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

## Version
v1.0, 2023/02/28

## Contact
Andrea Cherubini - acherubini@cosmoimd.com

## License
CC BY-NC-SA 4.0, https://creativecommons.org/licenses/by-nc-sa/4.0/

