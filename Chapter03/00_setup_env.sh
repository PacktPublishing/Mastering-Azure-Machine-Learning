#!/bin/bash
set -euo pipefail


# Check for existing Azure CLI instance
az -v 1> /dev/null

if [ $? != 0 ]; then
	echo "Azure CLI could not be found. Installing.."
	set -e
	(
		set -x
		# Install Azure CLI
        AZ_REPO=$(lsb_release -cs)
        echo "deb [arch=amd64] https://packages.microsoft.com/repos/azure-cli/ ${AZ_REPO} main" | sudo tee /etc/apt/sources.list.d/azure-cli.list
        curl -L https://packages.microsoft.com/keys/microsoft.asc | sudo apt-key add -
        sudo apt-get -y install apt-transport-https  1> /dev/null
        sudo apt-get -y update 1> /dev/null
        sudo apt-get -y install azure-cli
	)
	else
    AZ_VERSION=$(az -v |grep azure-cli | head -n1 | awk '{ print $2 }')
	echo "Using Azure CLI version ${AZ_VERSION}..."
fi


# Check for existing Azure CLI ML extension
az -v |grep azure-cli-ml 1> /dev/null

if [ $? != 0 ]; then
	echo "Azure CLI ML extension could not be found. Installing.."
	set -e
	(
		set -x
        # Install Azure CLI ML extension
        az extension add -n azure-cli-ml
    )
	else
    AZ_ML_VERSION=$(az -v | grep azure-cli-ml | head -n1 | awk '{ print $2 }')
	echo "Using Azure CLI ML extension version ${AZ_ML_VERSION}..."
fi


# Check for existing Pipenv installation
python3 -m pipenv --version 1> /dev/null

if [ $? != 0 ]; then
	echo "Pipenv could not be found. Installing.."
	set -e
	(
		set -x
        # Install Pipenv
        sudo apt-get -y install python3 python3-pip
	python3 -m pip install pipenv
	)
	else
    	PIPENV_VERSION=$(python3 -m pipenv --version | awk '{ print $3 }')
	echo "Using Pipenv version ${PIPENV_VERSION}..."
fi


# Install all required packages in virtual env
python3 -m pipenv install -r requirements.txt
