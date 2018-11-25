# High-resolution fingerprint recognition
This repository contains the original implementation of the fingerprint pore description models from "Automatic Dataset Annotation to Learn CNN Pore Description for Fingerprint Recognition".

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

## Pore description
### Generating pore identity annotations
It is required to have pore label annotations to train a pore descriptor. To do this, run:
```
python3 -m polyu.preprocess --polyu_dir_path polyu_hrf --pts_dir_path log/pores --patch_size 32 --result_dir_path log/patch_polyu
```
`log/pores` contains pore detections for every image in PolyU-HRF. This splits DBI Training in two subject independent subsets, one for training, another for validation. The default split creates the training subset with 60% of the subject identities. All other identites go to the validation subset.
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

## Fingerprint recogntion ablation study
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

To obtain exactly the same results as the paper, a trained model is provided in `log/description/augment_dropout-0.3`. To validate with it, then, run:
```
python3 -m validate.matching --polyu_dir_path polyu_hrf --pts_dir_path log/pores --descriptors trained --thr 0.7 --model_dir_path log/description/augment_dropout-0.3/bs-256_lr-1e-01_t-2018-08-20_083311 --fold DBI-test --patch_size 32
```
and
```
python3 -m validate.matching --polyu_dir_path polyu_hrf --pts_dir_path log/pores --descriptors trained --thr 0.7 --model_dir_path log/description/augment_dropout-0.3/bs-256_lr-1e-01_t-2018-08-20_083311 --fold DBII --patch_size 32
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
