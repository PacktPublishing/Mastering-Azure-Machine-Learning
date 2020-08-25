#!/bin/bash
set -euo pipefail

EXAMPLES_DIR=examples
PYSPARK="python3.6 -m pipenv run spark-submit"

JAR_AS=http://central.maven.org/maven2/com/microsoft/azure/azure-storage/8.3.0/azure-storage-8.3.0.jar
JAR_HA=http://central.maven.org/maven2/org/apache/hadoop/hadoop-azure/2.7.3/hadoop-azure-2.7.3.jar
JARS=jars

#login to azure using your credentials
az account show 1> /dev/null

if [ $? != 0 ];
then
	az login
fi

# Output commands
set -x

# Download the Azure-relevant jars
wget -N "${JAR_HA}" -P jars
wget -N "${JAR_AS}" -P jars

# Assemble the jars parameter
JARS=$(find "$(pwd)/jars" -name *.jar | tr '\n' ':' | sed 's/:$/\n/')

# Add jars to classpath to access blob via WASBS protocol
# https://stackoverflow.com/questions/37132559/add-jars-to-a-spark-job-spark-submit/37348234#37348234
$PYSPARK --driver-class-path "${JARS}" --conf "spark.executor.extraClassPath=${JARS}" \
    "${EXAMPLES_DIR}/load_dataset_to_spark.py"