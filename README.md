# High-resolution fingerprint recognition
This repository contains the original implementation of the fingerprint pore detection and description models from [CNN-based Pore Detection and Description for High-Resolution Fingerprint Recognition]() (Submitted to WACV 2019).

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

We recomend installing Python's virtualenv (tested for version 15.0.1) to run the experiments. To do it in Ubuntu 16.04:
```
sudo apt install python3-virtualenv
```

Then, create and activate a virtualenv:
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

### Validating trained detection models
To evaluate the model trained above, run:
```
python3 -m validate.detection --polyu_dir_path polyu_hrf --model_dir_path log/detection/[det_model_dir] --fold test --discard
```

The options for validating the detection model are:
```
usage: validate.detection [-h] --polyu_dir_path POLYU_DIR_PATH --model_dir_path
                          MODEL_DIR_PATH [--patch_size PATCH_SIZE] [--fold FOLD]
                          [--discard] [--results_path RESULTS_PATH] [--seed SEED]

optional arguments:
  -h, --help            show this help message and exit
  --polyu_dir_path POLYU_DIR_PATH
                        path to PolyU-HRF dataset
  --model_dir_path MODEL_DIR_PATH
                        logging directory
  --patch_size PATCH_SIZE
                        pore patch size
  --fold FOLD           which fold of the dataset to use. Can be "train",
                        "val", or "test"
  --discard             use this flag to disconsider pores in ground truth
                        borders
  --results_path RESULTS_PATH
                        path in which to save results
  --seed SEED           random seed
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
