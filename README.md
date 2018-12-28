# High-resolution fingerprint recognition
This repository contains the original implementation of the fingerprint pore detection and description models from [Automatic Dataset Annotation to Learn CNN Pore Description for Fingerprint Recognition](https://arxiv.org/abs/1809.10229).

## PolyU-HRF dataset
The Hong Kong Polytechnic University (PolyU) High-Resolution-Fingerprint (HRF) Database is a high-resolution fingerprint dataset for fingerprint recognition. We ran all of our experiments in the PolyU-HRF dataset, so it is required to reproduce them. PolyU-HRF can be obtained by following the instructions from its authors [here](http://www4.comp.polyu.edu.hk/~biometrics/HRF/HRF_old.htm).

Assuming PolyU-HRF is inside a local directory named `polyu_hrf`, its internal organization must be as following in order to reproduce our experiments with the code in this repository as it is:
```
polyu_hrf/
  DBI/
    Training/
    Test/
  DBII/
  GroundTruth/
    PoreGroundTruth/
      PoreGroundTruthMarked/
      PoreGroundTruthSampleimage/
```

## Requirements
The code in this repository was tested for Ubuntu 16.04 and Python 3.5.2, but we believe any newer version of both will do.

We recomend installing Python's venv (tested for version 15.0.1) to run the experiments. To do it in Ubuntu 16.04:
```
sudo apt install python3-venv
```

Then, create and activate a venv:
```
python3 -m venv env
source env/bin/activate
```

To install the requirements either run, for CPU usage:
```
pip install -r cpu-requirements.txt
```
or run, for GPU usage, which requires the [Tensorflow GPU dependencies](https://www.tensorflow.org/install/gpu):
```
pip install -r gpu-requirements.txt
```

## Pore detection
### Training the model
Throught our experiments, we will assume that PolyU-HRF is inside a local folder name `polyu_hrf`. To train a pore detection network with our best found parameters, run:
```
python3 -m train.detection --polyu_dir_path polyu_hrf --log_dir_path log/detection --dropout 0.5 --augment
```
This will create a folder inside `log/detection` for the trained model's resources. We will call it `[det_model_dir]` for the rest of the instructions.

The options for training the detection net are:
```
usage: train.detection [-h] --polyu_dir_path POLYU_DIR_PATH
                       [--learning_rate LEARNING_RATE]
                       [--log_dir_path LOG_DIR_PATH] [--dropout DROPOUT]
                       [--augment] [--tolerance TOLERANCE]
                       [--batch_size BATCH_SIZE] [--steps STEPS]
                       [--label_size LABEL_SIZE] [--label_mode LABEL_MODE]
                       [--patch_size PATCH_SIZE] [--seed SEED]
                       
optional arguments:
  -h, --help            show this help message and exit
  --polyu_dir_path POLYU_DIR_PATH
                        path to PolyU-HRF dataset
  --learning_rate LEARNING_RATE
                        learning rate
  --log_dir_path LOG_DIR_PATH
                        logging directory
  --dropout DROPOUT     dropout rate in last convolutional layer
  --augment             use this flag to perform dataset augmentation
  --tolerance TOLERANCE
                        early stopping tolerance
  --batch_size BATCH_SIZE
                        batch size
  --steps STEPS         maximum training steps
  --label_size LABEL_SIZE
                        pore label size
  --label_mode LABEL_MODE
                        how to convert pore coordinates into labels
  --patch_size PATCH_SIZE
                        pore patch size
  --seed SEED           random seed

```
for more details, refer to the code documentation.

### Validating the trained model
To evaluate the model trained above, run:
```
python3 -m validate.detection --polyu_dir_path polyu_hrf --model_dir_path log/detection/[det_model_dir]
```
The results will most likely differ from the ones reported in the paper. To reproduce those, read below about the trained models.

The options for validating the detection model are:
```
usage: validate.detection [-h] --polyu_dir_path POLYU_DIR_PATH --model_dir_path
                          MODEL_DIR_PATH [--patch_size PATCH_SIZE]
                          [--results_path RESULTS_PATH] [--seed SEED]

optional arguments:
  -h, --help            show this help message and exit
  --polyu_dir_path POLYU_DIR_PATH
                        path to PolyU-HRF dataset
  --model_dir_path MODEL_DIR_PATH
                        logging directory
  --patch_size PATCH_SIZE
                        pore patch size
  --results_path RESULTS_PATH
                        path in which to save results
  --seed SEED           random seed

usage: validate.detection [-h] --polyu_dir_path POLYU_DIR_PATH --model_dir_path
                          MODEL_DIR_PATH [--patch_size PATCH_SIZE] [--fold FOLD]
                          [--results_path RESULTS_PATH] [--seed SEED]
```

## Pore description
### Detecing pores in every image
To train a pore descriptor, pore detections are required for every image. To do this, run:
```
python3 -m batch_detect_pores --polyu_dir_path polyu_hrf --model_dir_path log/detection/[det_model_dir] --results_dir_path log/pores
```
This will detect pores for every image in PolyU-HRF and store them in `[image_name].txt` format inside `log/pores` subfolders `DBI/Training`, `DBI/Test` and `DBII`.

The options for batch detecting pores are:
```
usage: batch_detect_pores [-h] --polyu_dir_path POLYU_DIR_PATH
                          --model_dir_path MODEL_DIR_PATH
                          [--patch_size PATCH_SIZE]
                          [--results_dir_path RESULTS_DIR_PATH]
                          [--prob_thr PROB_THR] [--inter_thr INTER_THR]

optional arguments:
  -h, --help            show this help message and exit
  --polyu_dir_path POLYU_DIR_PATH
                        path to PolyU-HRF dataset
  --model_dir_path MODEL_DIR_PATH
                        path from which to restore trained model
  --patch_size PATCH_SIZE
                        pore patch size
  --results_dir_path RESULTS_DIR_PATH
                        path to folder in which results should be saved
  --prob_thr PROB_THR   probability threshold to filter detections
  --inter_thr INTER_THR
                        nms intersection threshold

```

### Generating pore identity annotations
It is also required to have pore identity annotations to train a pore descriptor. To do this, run:
```
python3 -m polyu.preprocess --polyu_dir_path polyu_hrf --pts_dir_path log/pores --patch_size 32 --result_dir_path log/patch_polyu
```
This splits DBI Training in two subject independent subsets, one for training, another for validation. The default split creates the training subset with 60% of the subject identities. All other identites go to the validation subset.
`polyu.preprocess` separates these subsets in subfolders of `log/patch_polyu`: `train` and `val`.

`train` contains pore patch images named `pore-id_register-id.png`. It has exactly 6 images for each pore identity that is visible in every image of the training subset.

`val` contains the images of the validation subset, with their original names, and their corresponding pore detection files.

The options for generating pore identity annotations are:
```
usage: polyu.preprocess [-h] --polyu_dir_path POLYU_DIR_PATH --pts_dir_path
                        PTS_DIR_PATH --patch_size PATCH_SIZE --result_dir_path
                        RESULT_DIR_PATH [--split SPLIT] [--seed SEED]

optional arguments:
  -h, --help            show this help message and exit
  --polyu_dir_path POLYU_DIR_PATH
                        path to PolyU-HRF dataset
  --pts_dir_path PTS_DIR_PATH
                        path to PolyU-HRF DBI Training dataset keypoints
                        detections
  --patch_size PATCH_SIZE
                        image patch size for descriptor
  --result_dir_path RESULT_DIR_PATH
                        path to save description dataset
  --split SPLIT         floating point percentage of training set in train/val
                        split
  --seed SEED           random seed
```

### Training the model
To train the pore description model, run:
```
python3 -m train.description --dataset_path log/patch_polyu --log_dir_path log/description/ --augment --dropout 0.3
```
This will train a description model with the hyper-parameters we used for the model in our paper, but we recommend tuning them manually by observing the EER in the validation set. The above values usually provide excelent results. However, if the model fails to achieve 0% EER in the validation set, you should probably investigate other values. Training without augmentation has disastrous results, so always train with it.

Running the script above will create a folder inside `log/description` for the trained model's resources. We will call it `[desc_model_dir]` for the rest of the instructions.

Options for training the description model are:
```
usage: train.description [-h] --dataset_path DATASET_PATH
                         [--learning_rate LEARNING_RATE]
                         [--log_dir_path LOG_DIR_PATH] [--tolerance TOLERANCE]
                         [--batch_size BATCH_SIZE] [--steps STEPS] [--augment]
                         [--dropout DROPOUT] [--weight_decay WEIGHT_DECAY]
                         [--seed SEED]

optional arguments:
  -h, --help            show this help message and exit
  --dataset_path DATASET_PATH
                        path to dataset
  --learning_rate LEARNING_RATE
                        learning rate
  --log_dir_path LOG_DIR_PATH
                        logging directory
  --tolerance TOLERANCE
                        early stopping tolerance
  --batch_size BATCH_SIZE
                        batch size
  --steps STEPS         maximum training steps
  --augment             use this flag to perform dataset augmentation
  --dropout DROPOUT     dropout rate in last convolutional layer
  --weight_decay WEIGHT_DECAY
                        weight decay lambda
  --seed SEED           random seed
```

## Fingerprint recogntion experiments
### SIFT descriptors
In order to reproduce the SIFT descriptors experiment in DBI Test, run:
```
python3 -m validate.matching --polyu_dir_path polyu_hrf --pts_dir_path log/pores --descriptors sift --thr 0.7 --fold DBI-test
```
For the DBII experiment:
```
python3 -m validate.matching --polyu_dir_path polyu_hrf --pts_dir_path log/pores --descriptors sift --thr 0.7 --fold DBII
```

### DP descriptors
For the DBI Test, run:
```
python3 -m validate.matching --polyu_dir_path polyu_hrf --pts_dir_path log/pores --descriptors dp --thr 0.7 --fold DBI-test --patch_size 32
```
For the DBII experiment:
```
python3 -m validate.matching --polyu_dir_path polyu_hrf --pts_dir_path log/pores --descriptors dp --thr 0.7 --fold DBII --patch_size 32
```

### Trained descriptors
For the DBI Test experiment:
```
python3 -m validate.matching --polyu_dir_path polyu_hrf --pts_dir_path log/pores --descriptors trained --thr 0.7 --model_dir_path log/description/[desc_model_dir] --fold DBI-test --patch_size 32
```
For the DBII experiment:
```
python3 -m validate.matching --polyu_dir_path polyu_hrf --pts_dir_path log/pores --descriptors trained --thr 0.7 --model_dir_path log/description/[desc_model_dir] --fold DBII --patch_size 32
```

Other options for `validate.matching` are:
```
usage: validate.matching [-h] --polyu_dir_path POLYU_DIR_PATH --pts_dir_path
                         PTS_DIR_PATH [--results_path RESULTS_PATH]
                         [--descriptors DESCRIPTORS] [--mode MODE] [--thr THR]
                         [--model_dir_path MODEL_DIR_PATH] [--patch_size PATCH_SIZE]
                         [--fold FOLD] [--seed SEED]

optional arguments:
  -h, --help            show this help message and exit
  --polyu_dir_path POLYU_DIR_PATH
                        path to PolyU-HRF dataset
  --pts_dir_path PTS_DIR_PATH
                        path to chosen dataset keypoints detections
  --results_path RESULTS_PATH
                        path to results file
  --descriptors DESCRIPTORS
                        which descriptors to use. Can be "sift", "dp" or
                        "trained"
  --mode MODE           mode to match images. Can be "basic" or "spatial"
  --thr THR             distance ratio check threshold
  --model_dir_path MODEL_DIR_PATH
                        trained model directory path
  --patch_size PATCH_SIZE
                        pore patch size
  --fold FOLD           choose what fold of polyu to use. Can be "DBI-train",
                        "DBI-test" and "DBII"
  --seed SEED           random seed
```

## Pre-trained models and reproducing paper results
The pre-trained [detection](https://drive.google.com/open?id=1U9rm_5za2kRU2FsviCe-qrZoouwUGyzI) and [description](https://drive.google.com/open?id=16GiLG7xBj64SOjCJwlCfbBcb-DORzYg1) models are required to ensure that you get the exact same results as those of the paper. After downloading them, follow the batch detection and fingerprint recognition steps replacing `[det_model_dir]` and `[desc_model_dir]` where appropriate.

## Recognizing fingerprints
We also provide `recognize.py`, a script to, given two high resolution fingerprint images and a model trained to detect pores and another one to describe them, determine if they are from the same subject or not. To use it, run:
```
python3 -m recognize --image_paths [image01_path] [image02_path] --det_model_dir [det_model_dir] --desc_model_dir [desc_model_dir]
```

There is also a command line parameter, `score_thr`, to control the minimum number of established correspondences to determine that the images belong to the same subject. Its default value is 2, the EER threshold for the partial fingerprints in DBI-test. For the full fingerprints of DBII, this value should be set to 9.

Other options for this script are:
```
usage: recognize [-h] --img_paths IMG_PATHS IMG_PATHS --det_model_dir
                 DET_MODEL_DIR --desc_model_dir DESC_MODEL_DIR
                 [--score_thr SCORE_THR] [--det_patch_size DET_PATCH_SIZE]
                 [--det_prob_thr DET_PROB_THR]
                 [--nms_inter_thr NMS_INTER_THR]
                 [--desc_patch_size DESC_PATCH_SIZE]

optional arguments:
  -h, --help            show this help message and exit
  --img_paths IMG_PATHS IMG_PATHS
                        path to images to be recognized
  --det_model_dir DET_MODEL_DIR
                        path to pore detection trained model
  --desc_model_dir DESC_MODEL_DIR
                        path to pore description trained model
  --score_thr SCORE_THR
                        score threshold to determine if pair is genuine or
                        impostor
  --det_patch_size DET_PATCH_SIZE
                        detection patch size
  --det_prob_thr DET_PROB_THR
                        probability threshold for discarding detections
  --nms_inter_thr NMS_INTER_THR
                        NMS area intersection threshold
  --desc_patch_size DESC_PATCH_SIZE
                        patch size around each detected keypoint to describe
```

## Reference
If you find the code in this repository useful for your research, please consider citing:
```
@article{dahia2018cnn,
  title={Automatic Dataset Annotation to Learn CNN Pore Description for Fingerprint Recognition},
  author={Dahia, Gabriel and Segundo, Maur{\'\i}cio Pamplona},
  journal={arXiv preprint arXiv:1809.10229},
  year={2018}
}
```

## License
See the [LICENSE](LICENSE) file for details.
