# REAL-Colon Dataset

## Description
The REAL (Real-world multi-center Endoscopy Annotated video Library) - Colon dataset comprises 60 real-world colonoscopy recordings collected from six centers across the globe. These recordings originate from four distinct clinical studies (001 to 004), with each study contributing 15 videos.

For each patient/video, various clinical variables were collected, including the endoscope brand, bowel cleanliness score (BBPS), the number of surgically removed colon lesions, as well as the patient's age and sex. Trained labelers, under the supervision of expert gastroenterologists, annotated each polyp by placing a bounding box around it in every frame where it appeared. Additionally, information on each polyp, such as histology, size, and anatomical site, was recorded.

For complete details about the dataset and to cite this work, please refer to [1].

### Key Statistics
- **Recordings:** 60 (15 recordings per clinical study)
- **Total Frames:** 2,757,723
- **Removed Colorectal Polyps:** 132
- **Bounding Box Annotations:** 351,264

## Data Format
The dataset includes the following files:

- **60 compressed folders** named `{SSS}-{VVV}_frames` containing the frames for each recording.
- **60 compressed folders** named `{SSS}-{VVV}_annotation` containing the annotations for each recording.
- **video_info.csv**: A file containing metadata for each video.
- **lesion_info.csv**: A file containing metadata for each lesion.
- **dataset_description.md**: A README file providing detailed information about the dataset.

### Annotation File
Each XML file follows this format:

**Sample VOC Notation:**
```xml
<annotation>
    <version_fmt>1.0</version_fmt>
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
        <box_id>1</box_id> <!-- id of the box within the image -->
        <bndbox>
            <xmin>540</xmin>
            <xmax>1196</xmax>
            <ymin>852</ymin>
            <ymax>1070</ymax>
        </bndbox>
    </object>
</annotation>
```

### CSV Files
The two CSV files contain metadata for each video (`video_info.csv`) and each lesion (`lesion_info.csv`).

The `video_info.csv` file contains the following data:
- `unique_video_name`: Name of the video in the format SSS-VVV (site-video)
- `age`: Patient's age
- `sex`: Patient's sex
- `endoscope_brand`: Endoscope brand (Fuji or Olympus)
- `fps`: Video frames per second
- `num_frames`: Video length in frames
- `num_lesions`: Number of removed lesions
- `bbps`: Boston Bowel Preparation Score

The `lesion_info.csv` file contains the following data:
- `unique_object_id`: Lesion ID in the format `unique_video_name_x`
- `unique_video_name`: Name of the video to which the lesion belongs
- `size [mm]`: Lesion size in millimeters
- `site`: Anatomical site where the removed lesion was found
- `histology_extended`: Detailed histology of the lesion
- `histology_class`: Lesion histology assigned to one of the following classes: (adenoma, AD; hyperplastic polyp, HP; sessile serrated lesion, SSL; NO POLYP, traditional serrated adenoma, TSA; OTHER)

## Dataset Download
Run `figshare_dataset.py` to automatically download the entire dataset from Figshare to the `./dataset` folder. The output folder can be changed by updating the `DOWNLOAD_DIR` variable in `figshare_dataset.py`. The dataset contains roughly 1TB of data and, depending on your internet connection and hardware, it may take several days to download. Please plan accordingly.

The dataset is available on Figshare: [https://plus.figshare.com/articles/media/REAL-colon_dataset/22202866](https://plus.figshare.com/articles/media/REAL-colon_dataset/22202866) and archived at this DOI: [10.25452/figshare.plus.22202866](https://doi.org/10.25452/figshare.plus.22202866).

## Dataset Exploration and Stats
To visualize random example images from the dataset, along with associated clinical information, you can use the Jupyter Notebook `explore_data.ipynb`.

To reproduce the statistics and plots reported in [1], please refer to the Python scripts `dataset_stats.py` and `bounding_boxes_plots.py`.

## Polyp Detection
To convert the data into the COCO format for polyp detection model training and validation, and to reproduce the polyp detection results presented in [1], please refer to the: [Polyp Detection](polyp_detection/README.md) page.

## References
[1] Biffi, C., Antonelli, G., Bernhofer, S., Hassan, C., Hirata, D., Iwatate, M., Maieron, A., Salvagnini, P., & Cherubini, A. (2024). REAL-Colon: A dataset for developing real-world AI applications in colonoscopy. Scientific Data, 11(1), 539. https://doi.org/10.1038/s41597-024-03359-0

## Contact
For any inquiries, please open an issue in this repository or contact us directly at:
- **Carlo Biffi** - cbiffi@cosmoimd.com
- **Andrea Cherubini** - acherubini@cosmoimd.com

## License
CC BY 4.0, [https://creativecommons.org/licenses/by/4.0/](https://creativecommons.org/licenses/by/4.0/)

