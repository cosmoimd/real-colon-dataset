# REAL-colon dataset

The following instructions allow REAL-Colon data to be formatted in the COCO format and used to train object detectors as defined for example in https://github.com/cosmoimd/DeepLearningExamples/tree/master/PyTorch/Detection

### Convert Images and Annotations to COCO Format
Use `export_coco_format.py` to create from the downloaded REAL-Colon dataset a dataset with the number of images per polyp and negative frames per video for train/valid/test splits:
- Set `base_dataset_folder` to your dataset's location.
- Set `output_folder` for the COCO-formatted output.

### Additional Testing splits
Given the output of `export_coco_format.py` in `base_dataset_folder`  use `filter_annotation.py` to create additional test sublist from the test annotations.

## SSD Model Training and Evaluation
### Training
To train a model defined in https://github.com/cosmoimd/DeepLearningExamples/tree/master/PyTorch/Detection/SSD:
- Build and run the docker container with `docker build . -t nvidia_ssd` and then `docker run --rm -it --gpus=all --ipc=host nvidia_ssd`. Here you can also
add any paths necessary for the code using the -v flag.
- Add to the `dataset_folder` in `PyTorch/Detection/SSD/ssd/utils.py` the `output_folder` path obtained from the `export_coco_format.py` code run in the previous step. In this way, the model will be trained with an user-defined train/valid/test split of the data according to the user needds. 
- To start training run: `CUDA_VISIBLE_DEVICES=0 python main.py --dataset-name real_colon --backbone resnet50 --warmup 300 --bs 64 --epochs 65 --data /coco --save ./models`.
This will also save the model checkpoint in `./models`.

### Validation
To evaluate the trained models:
- In the docker container, run `python ./main.py --backbone resnet50 --dataset-name real_colon 
--json-save-path /path/to/save/json/files --mode testing --no-skip-empty --checkpoint /your/model/path --data /path/to/dir/containing/test/set/`
