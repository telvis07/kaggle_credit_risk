#!/usr/bin/env bash
set -e
set -v

RESULTS_DIR="./results"
MODELS_DIRS="./models"

#rm -rf ${RESULTS_DIR}
mkdir -p ${RESULTS_DIR}
mkdir -p ${MODELS_DIRS}

RESULTS_JSON="${RESULTS_DIR}/main_only.json"
RESULTS_PNG="${RESULTS_DIR}/main_only.png"
OUTPUT_MODEL="${MODELS_DIRS}/main_only}"
INPUT_CSV="./merged_datasets/main_only/train.csv.gz"

if ! [[ -f ${RESULTS_JSON} ]]; then
    python dnn.py ${INPUT_CSV} \
      -m ${OUTPUT_MODEL} \
      -o ${RESULTS_JSON} \
      -p ${RESULTS_PNG}
fi


RESULTS_JSON="${RESULTS_DIR}/main_bureau.json"
RESULTS_PNG="${RESULTS_DIR}/main_bureau.png"
OUTPUT_MODEL="${MODELS_DIRS}/main_bureau"
INPUT_CSV="./merged_datasets/main_bureau/train.csv.gz"


if ! [[ -f ${RESULTS_JSON} ]]; then
    python dnn.py ${INPUT_CSV} \
      -m ${OUTPUT_MODEL} \
      -o ${RESULTS_JSON} \
      -p ${RESULTS_PNG}
fi



RESULTS_JSON="${RESULTS_DIR}/main_bureau_poscash.json"
RESULTS_PNG="${RESULTS_DIR}/main_bureau_poscash.png"
OUTPUT_MODEL="${MODELS_DIRS}/main_bureau_poscash"
INPUT_CSV="./merged_datasets/main_bureau_poscash/train.csv.gz"

if ! [[ -f ${RESULTS_JSON} ]]; then
    python dnn.py ${INPUT_CSV} \
      -m ${OUTPUT_MODEL} \
      -o ${RESULTS_JSON} \
      -p ${RESULTS_PNG}
fi



RESULTS_JSON="${RESULTS_DIR}/main_bureau_ccbal.json"
RESULTS_PNG="${RESULTS_DIR}/main_bureau_ccbal.png"
OUTPUT_MODEL="${MODELS_DIRS}/main_bureau_ccbal"
INPUT_CSV="./merged_datasets/main_bureau_ccbal/train.csv.gz"

if ! [[ -f ${RESULTS_JSON} ]]; then
    python dnn.py ${INPUT_CSV} \
      -m ${OUTPUT_MODEL} \
      -o ${RESULTS_JSON} \
      -p ${RESULTS_PNG}
fi


RESULTS_JSON="${RESULTS_DIR}/main_bureau_prevapp.json"
RESULTS_PNG="${RESULTS_DIR}/main_bureau_prevapp.png"
OUTPUT_MODEL="${MODELS_DIRS}/main_bureau_prevapp"
INPUT_CSV="./merged_datasets/main_bureau_prevapp/train.csv.gz"

if ! [[ -f ${RESULTS_JSON} ]]; then
    python dnn.py ${INPUT_CSV} \
      -m ${OUTPUT_MODEL} \
      -o ${RESULTS_JSON} \
      -p ${RESULTS_PNG}
fi


RESULTS_JSON="${RESULTS_DIR}/main_bureau_instpay.json"
RESULTS_PNG="${RESULTS_DIR}/main_bureau_instpay.png"
OUTPUT_MODEL="${MODELS_DIRS}/main_bureau_instpay"
INPUT_CSV="./merged_datasets/main_bureau_instpay/train.csv.gz"

if ! [[ -f ${RESULTS_JSON} ]]; then
    python dnn.py ${INPUT_CSV} \
      -m ${OUTPUT_MODEL} \
      -o ${RESULTS_JSON} \
      -p ${RESULTS_PNG}
fi



RESULTS_JSON="${RESULTS_DIR}/main_plus_all_numeric_files.json"
RESULTS_PNG="${RESULTS_DIR}/main_plus_all_numeric_files.png"
OUTPUT_MODEL="${MODELS_DIRS}/main_plus_all_numeric_files"
INPUT_CSV="./merged_datasets/main_plus_all_numeric_files/train.csv.gz"

if ! [[ -f ${RESULTS_JSON} ]]; then
    python dnn.py ${INPUT_CSV} \
      -m ${OUTPUT_MODEL} \
      -o ${RESULTS_JSON} \
      -p ${RESULTS_PNG}
fi


RESULTS_JSON="${RESULTS_DIR}/main_plus_all_label_files.json"
RESULTS_PNG="${RESULTS_DIR}/main_plus_all_label_files.png"
OUTPUT_MODEL="${MODELS_DIRS}/main_plus_all_label_files"
INPUT_CSV="./merged_datasets/main_plus_all_label_files/train.csv.gz"

if ! [[ -f ${RESULTS_JSON} ]]; then
    python dnn.py ${INPUT_CSV} \
      -m ${OUTPUT_MODEL} \
      -o ${RESULTS_JSON} \
      -p ${RESULTS_PNG}
fi


RESULTS_JSON="${RESULTS_DIR}/main_plus_all.json"
RESULTS_PNG="${RESULTS_DIR}/main_plus_all.png"
OUTPUT_MODEL="${MODELS_DIRS}/main_plus_all"
INPUT_CSV="./merged_datasets/main_plus_all/train.csv.gz"

if ! [[ -f ${RESULTS_JSON} ]]; then
    python dnn.py ${INPUT_CSV} \
      -m ${OUTPUT_MODEL} \
      -o ${RESULTS_JSON} \
      -p ${RESULTS_PNG}
fi


