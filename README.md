# registration-da-mean-teacher
Source code for our Miccai2022 paper 'Domain adaptive keypoint-based lung registration with the Mean Teacher'.

## Dependencies
Please first install the following dependencies
* Python3 (we use 3.9.7)
* numpy
* pytorch (we use 1.10.2)

## Data
We provide the Foerstner keypoint that we used for training and inference in `data.pth`.
Thus, downloading or pre-processing data is not required to repeat our experiments.

## Training
Execute `python main.py --gpu GPU --phase train --method {ours, source, target} --setting {copd_to_l2r, 4dct_to_copd} --out_dir path/to/out_dir`.
This will train the specified method (ours, source-ony, or target-only) under the specified adaptation scenario (copd_to_l2r or 4dct_to_copd) and write model parameters and log file to the specified out_dir.

## Testing
Execute `python main.py --gpu GPU --phase test --setting {copd_to_l2r, 4dct_to_copd} --model_path path/to/model --out_dir path/to/out_dir`
This will do inference under the specified adaptation scenario (copd_to_l2r or 4dct_to_copd), using the specified model weights.
For 4dct_to_copd, this will directly print the resulting TRE [mm] on the COPD dataset.
For copd_to_l2r, this will save predicted dense displacement fields (interpolated from sparse displacement vectors) to the specified output directory.
They are in the correct format for upload to the official [Learn2Reg evaluation server](https://learn2reg.grand-challenge.org/Submission/), which will output the TRE of the predictions.

You can either use your own trained models or our pre-trained models, provided at `models`.
In the latter case, please ensure that the selected model matches the selected adaptation scenario.
For our model and the target-only under the 4dct_to_copd, please specify the model path as `models/4dct_to_copd_ours_fold{}.pth` or `models/4dct_to_copd_target-only_fold{}.pth`
Models of the appropriate folds will be selected automatically.
Using our models should reproduce the numerical results presented in Table 1 of our paper.
