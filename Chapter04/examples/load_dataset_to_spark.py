from azureml.core.authentication import AzureCliAuthentication
from azureml.core.workspace import Workspace
from azureml.core.datastore import Datastore
from azureml.core.dataset import Dataset
from pyspark import SparkContext, SparkConf

dstore_name = 'mldemodatastore'
ds_file = "movielens100k.movies"

# Configure workspace
cli_auth = AzureCliAuthentication()
ws = Workspace.from_config(auth=cli_auth)

# Access your dataset
dataset = Dataset.get(ws, ds_file)

# Load in-memory Dataset to your local machine as pandas dataframe
sparkDf = dataset.to_spark_dataframe()
sparkDf.show()
