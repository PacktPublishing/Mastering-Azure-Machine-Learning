#!/bin/bash
set -euo pipefail

EXAMPLES_DIR=examples
PYTHON="python3.6 -m pipenv run python"

#login to azure using your credentials
az account show 1> /dev/null

if [ $? != 0 ];
then
	az login
fi

# Output commands
set -x

# Load and register dataset in Python
$PYTHON "${EXAMPLES_DIR}/load_dataset_to_pandas.py"