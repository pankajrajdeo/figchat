#!/bin/bash -x

PORT=$1
[ -z "$PORT" ] && PORT=7860
[ $PORT -lt 1000 ] && ROOT_PATH_POSTFIX=$PORT/ && PORT=$((7860+PORT))/

. ../OPENAI_API_KEY.figchat
. ../GOOGLE_API_KEY.figchat
. ../CHAINLIT_AUTH_SECRET.figchat

BASE_DATASET_DIR=/reference-data/figchat_datasets/
PLOT_OUTPUT_DIR=/reference-data/figchat_plots/
DATASET_INDEX_FILE=/reference-data/figchat_datasets/dataset_index_advanced_paths.tsv
DATABASE_URL=postgresql://chainlit:myChain3@chainlit.cbijlsssksnt.us-east-1.rds.amazonaws.com/postgres
TRAIN_DATA_FILE=/reference-data/figchat_datasets/TRAIN_DATA_FILE.json
TRAIN_IMAGE_DATA_FILE=/reference-data/figchat_datasets/TRAIN_IMAGE_DATA_FILE.json
BASE_URL=https://devapp.lungmap.net
ROOT_PATH=/figchat${ROOT_PATH_POSTFIX}

docker run -d --restart unless-stopped -p $PORT:7860  \
	-v $BASE_DATASET_DIR:$BASE_DATASET_DIR \
	-v $PLOT_OUTPUT_DIR:$PLOT_OUTPUT_DIR \
        -e BASE_DATASET_DIR="$BASE_DATASET_DIR" \
        -e PLOT_OUTPUT_DIR="$PLOT_OUTPUT_DIR" \
        -e DATABASE_URL="$DATABASE_URL" \
        -e DATASET_INDEX_FILE="$DATASET_INDEX_FILE" \
        -e TRAIN_DATA_FILE="$TRAIN_DATA_FILE" \
        -e TRAIN_IMAGE_DATA_FILE="$TRAIN_IMAGE_DATA_FILE" \
        -e OPENAI_API_KEY="$OPENAI_API_KEY" \
        -e BASE_URL="$BASE_URL" \
        -e CHAINLIT_AUTH_SECRET="$CHAINLIT_AUTH_SECRET" \
        -e GOOGLE_API_KEY="$GOOGLE_API_KEY" \
	--name figchat \
	figchat --root-path $ROOT_PATH

