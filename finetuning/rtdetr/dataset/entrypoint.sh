#!/bin/bash

set -x

if [[ "$1" == 'generate_prelabels' ]];then


    python generate_dataset_bbox.py "/home/frank/datasets/examenes_de_admision/publicas/UNMSM - Universidad Nacional Mayor de San Marcos/2025-I"
    python export_images.py "/home/frank/Code/multumbabel/lab/finetuning/rtdetr/dataset/results_unmsm_2025.parquet"

fi

if [[ "$1" == 'run_label' ]];then

    echo "Starting Label Studio"
    export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
    export LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=/home/frank/Code/multumbabel/lab/finetuning/rtdetr/dataset

    label-studio start --username admin@admin.com --password admin 

fi


if [[ "$1" == 'upload_labels' ]];then

    echo "Uploading labels"
    python upload_annotations.py

fi

if [[ "$1" == 'download_labels' ]];then

    echo "Downloading labels"
    python download_annotations.py

fi