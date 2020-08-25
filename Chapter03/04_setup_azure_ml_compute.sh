#!/bin/bash
set -euo pipefail

RG=mldemo
LOC=westeurope
WS=mldemows

COMPUTE_NAME=amldemocompute

MIN_NODES=0
MAX_NODES=2
NODE_TYPE=STANDARD_D2_V2

#login to azure using your credentials
az account show 1> /dev/null

if [ $? != 0 ];
then
	az login
fi

# Output commands
set -x

# Create a new compute target
# docs: https://docs.microsoft.com/de-de/cli/azure/ext/azure-cli-ml/ml/computetarget/create?view=azure-cli-latest#ext-azure-cli-ml-az-ml-computetarget-create-amlcompute
az ml computetarget create amlcompute --name ${COMPUTE_NAME} --min-nodes ${MIN_NODES} --max-nodes ${MAX_NODES} -s ${NODE_TYPE}
