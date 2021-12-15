# Bigger\&Faster: Two-stage Neural Architecture Search for Quantized BERT Models

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

To create the latency dataset go to the script directory and edit the `lat_dataset_gen_quant.sh` to reflect the correct following variable paths:
PROJECT_ROOT:  path to the root of the Bigger&Faster repository \
SUPER_MODEL:  path to the super model \
SAVE_PATH_DIR: path to the where to save the dataset, including name of the dataset 

Now run `lat_dataset_gen_quant.sh`.

Note if you want to create a non quantized latency dataset simply follow the above steps but with `lat_dataset_gen.sh`.

### Step 5: Train latency predictor

To train the latency predictor go to the script directory and edit the `train_lat_predictor.sh` to reflect the correct following variable paths:

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

