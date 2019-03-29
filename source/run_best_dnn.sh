#!/usr/bin/env bash
set -e
set -v

RESULTS_DIR="./kaggle.submission"
#MODELS_DIRS="${RESULTS_DIR}/model"

rm -rf ${RESULTS_DIR}
mkdir -p ${RESULTS_DIR}

EXPERIMENT="main_plus_all_numeric_files"
RESULTS_JSON="${RESULTS_DIR}/${EXPERIMENT}/RESULTS.json"
RESULTS_PNG="${RESULTS_DIR}/${EXPERIMENT}/ROC.png"

# output directories
NPZDIR="${RESULTS_DIR}/${EXPERIMENT}/predictions"
mkdir -p ${NPZDIR}

OUTPUT_MODEL="${RESULTS_DIR}/${EXPERIMENT}/model"
mkdir -p ${OUTPUT_MODEL}

# training and testing csv
INPUT_CSV="./merged_datasets/${EXPERIMENT}/train.csv.gz"
TEST_CSV="./merged_datasets/${EXPERIMENT}/test.csv.gz"

KAGGLE_SUBMISSION_CSV="${RESULTS_DIR}/${EXPERIMENT}/submission.csv"


if ! [[ -f ${RESULTS_JSON} ]]; then
    python dnn.py ${INPUT_CSV} \
      -m ${OUTPUT_MODEL} \
      -o ${RESULTS_JSON} \
      -p ${RESULTS_PNG} \
      --npzdir ${NPZDIR} \
      --test_data ${TEST_CSV} \
      -k ${KAGGLE_SUBMISSION_CSV}
fi


