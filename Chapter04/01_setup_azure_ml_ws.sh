#!/bin/bash
set -euo pipefail

RG=packt-mastering-azure-machine-learning
LOC=westeurope
WS=packt
EX=mldemo

DN=packtmldemodatastore
SN=packtmldemoblob
SC=data

#login to azure using your credentials
az account show 1> /dev/null

if [ $? != 0 ];
then
	az login
fi

# Output commands
set -x

# Save ML workspace credentials
# docs: https://docs.microsoft.com/de-de/cli/azure/ext/azure-cli-ml/ml/folder?view=azure-cli-latest#ext-azure-cli-ml-az-ml-folder-attach
az ml folder attach -w ${WS} -g ${RG} -e ${EX}

# Create a blob storage account
# docs: https://docs.microsoft.com/en-us/cli/azure/storage/account?view=azure-cli-latest#az-storage-account-create
az storage account create -n ${SN} -g ${RG} --sku Standard_LRS --encryption-services blob

# Retrieve account key
ACCOUNT_KEY=$(az storage account keys list -n ${SN} -g ${RG} | jq '.[0].value')

# Create a container in the blob storage
# docs: https://docs.microsoft.com/en-us/cli/azure/storage/container?view=azure-cli-latest#az-storage-container-create
az storage container create -n ${SC} --account-name ${SN} --account-key ${ACCOUNT_KEY}

# Attach the blob container as data storage
# docs: https://docs.microsoft.com/de-de/cli/azure/ext/azure-cli-ml/ml/datastore?view=azure-cli-latest#ext-azure-cli-ml-az-ml-datastore-attach-blob
az ml datastore attach-blob -n ${DN} -a ${SN} -c ${SC} --account-key ${ACCOUNT_KEY}

