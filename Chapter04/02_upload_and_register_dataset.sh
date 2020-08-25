#!/bin/bash
set -euo pipefail

RG=packt-mastering-azure-machine-learning
LOC=westeurope
WS=packt
EX=mldemo

DN=packtmldemodatastore
SN=packtmldemoblob
SC=data

EXAMPLES_DIR=examples
DATA_DIR=data
PYTHON="python3.6 -m pipenv run python"

FILE_URL=http://files.grouplens.org/datasets/movielens/ml-latest-small.zip
DIR_NAME=ml-latest-small

#login to azure using your credentials
az account show 1> /dev/null

if [ $? != 0 ];
then
	az login
fi

# Output commands
set -x

# Download the dataset
wget -N "${FILE_URL}" -P "${DATA_DIR}"

# Unzipt the dataset
unzip -n "${DATA_DIR}/${DIR_NAME}.zip" -d "${DATA_DIR}"

# Retrieve account key
ACCOUNT_KEY=$(az storage account keys list -n ${SN} -g ${RG} | jq '.[0].value')

# Upload folder to blob storage
# docs: https://docs.microsoft.com/en-us/cli/azure/storage/blob?view=azure-cli-latest#az-storage-blob-upload-batch
az storage blob upload-batch \
    --account-name ${SN} \
    --account-key ${ACCOUNT_KEY} \
    --destination "${SC}/${DIR_NAME}" \
    --source "${DATA_DIR}/${DIR_NAME}"
