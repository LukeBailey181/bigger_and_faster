# Bigger\&Faster: Two-stage Neural Architecture Search for Quantized BERT Models

## Overview of repository

This repository is based off the AutoTinyBert repository as our project was an adaptation (and dare we say improvement) of the work they did. Here we give a brief overview of what each file does:

* `generate_data.py` - used for processing the raw BookCorpus dataset so it can be used to train the super model.
* `inference_time_evaluation_quant.py` - used to generate the quantized or unquantized latency predictor datasets. We incorporated PyTorch dynamic quantization into this file and edited how the search space was traversed. This is because we needed to covert between the transformer representation that AutoTinyBert used and that used in native HuggingFace so we could use PyTorch to quantize the models.
* `inference_time_evaluation_quant_onnx.py` - this file can also be used to create a quantized or unquantized latency predictor dataset but the quantization is done using onnx and onnxruntime. We did not end up using this method in our final work because the conversion to onnx was too slow.
* `latency_predictor.py` - contains the code to train the latency predictor and the latency predictor architecture.
* `pre_training.py` - used to train the super model.
* `quantize_utils.py` - custom int8 quantization utility functions we wrote to use during the search process.
* `script` - a directory that contains all the shell scripts we wrote to run various parts of the pipeline. More details on this in the below section about running our code.
* `seacher.py`, `superbert_run_en_classifier_fp.py` and, `superbert_run_en_classifier_ptq.py` - files to run the NAS process. These include additions we made to incorporate quantization into the NAS process.
* `Transformer` - edited version of the transformers library included in the AutoTinyBert repository. It contains BERT classes with more flexible architectures than vanilla HuggingFace.
  * `hf_converter.py` - a file we added to `Transformer` than can be used to convert between `Transformer` BERT models and HuggingFace models.
* `submodel_extractor.py` - used to extract submodels from super model.
* `utils.py` - architecture sampling utilities.

## How to run the code

### Step 1: Prepare environment

To begin with ensure you have python version 3.8 and create a virtual environment using your favourite Python environment handler (we recommend conda).
Once inside the environment, run the following to install all the required packages:

```
pip install -r requirements.txt
```

### Step 2: Generate dataset to train super model

Collect the raw data by following the steps in this repository: https://github.com/soskek/bookcorpus 

Go to the script directory and edit the `generate_data.sh` file to reflect the correct values for the following variables: 

PROJECT_ROOT:  path to the root of the Bigger&Faster repository \
RAW_DIR: path to the directory with the raw dataset in \
GENERATED_DIR: path to the directory to save the generated data 

Now run `generate_data.sh`.

### Step 3: Train super model 

Begin by downloading the starting model that will be trained from: https://pan.baidu.com/share/init?surl=uj8EuED2HeH6heMKAxHv_A 

Go to the script directory and edit the `pre_training.sh` file to reflect the correct values for the following variables:

PROJECT_ROOT:  path to the root of the Bigger&Faster repository \
GENERATED_DIR: path to the directory where the saved generated data is \
OUTPUT_DIR: output directory for trained super model \
STUDENT_MODEL: path to the directory where the model downloaded from baidu drive is 

### Step 4: Create latency dataset

To create the latency dataset go to the script directory and edit the `lat_dataset_gen_quant.sh` to reflect the correct values for the following variables:
PROJECT_ROOT:  path to the root of the Bigger&Faster repository \
SUPER_MODEL:  path to the super model \
SAVE_PATH_DIR: path to the where to save the dataset, including name of the dataset 

Now run `lat_dataset_gen_quant.sh`.

Note if you want to create a non quantized latency dataset simply follow the above steps but with `lat_dataset_gen.sh`.

### Step 5: Train latency predictor

To train the latency predictor go to the script directory and edit the `train_lat_predictor.sh` to reflect the correct values for the following variables:

PROJECT_ROOT: path to the root of the Bigger&Faster repository \
LAT_DATASET_PATH: path to the latency dataset \
PATH_TO_SAVE_MODEL: path to where you want to save the latency model including the name for the latency model 

Now run `train_lat_predictor.sh`.

### Step 6: Run NAS

First download the MNLI dataset from: https://gluebenchmark.com/tasks 

Go to the script directory and edit the `search.sh` file to reflect the correct values for the following variables:

PROJECT_ROOT: path to the root of the Bigger&Faster repository \
CKPT_PATH: path to quantized latency predictor \
MODEL: path to super model \
DATA_DIR: path to the "MNLI" directory that you downloaded from the glue benchmark website 

You will also notice a code block at the top that looks like:

```
if [ "$TYPE" = "fp" ]; then
CKPT_PATH="${PROJECT_ROOT}/conf_datasets/lat_predictor.pt"
fi
```

The CKPT_PATH here should be set to the path of your floating point latency predictor should you want to run the baseline search and not the quantized search.

Repeat the above for the `search_and_eval.sh` script.

Finally, go to the script directory and edit the `run_search.sh` file to reflect the correct values for the following variables:

TYPE: "ptq" for quantized search, "fp" for baseline floating point search \
LATENCY_CONSTRAINT: The latency constraint for the search 

Finally run `run_search.sh`. Results will be in a directory that looks like "output/kd_fp_10/subbert.results". The "kd-fp_10" will be different depending on what your search parameters are.

