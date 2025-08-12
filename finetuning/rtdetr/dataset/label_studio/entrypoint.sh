#!/bin/bash

export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
export LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=

label-studio start --username admin@admin.com --password admin
