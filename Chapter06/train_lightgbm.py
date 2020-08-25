import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgbm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import seaborn as sns
import umap.umap_ as umap
from sklearn.externals import joblib
sns.set(style='ticks', rc={'figure.figsize':(9,7)})


# Loading data
# ---------------------------------------
from azureml.core import Dataset, Run

# Load the current run and ws
run = Run.get_context()
ws = run.experiment.workspace

# Get a dataset by name
dataset = Dataset.get_by_name(workspace=ws, name='titanic_cleaned')

# Load a TabularDataset into pandas DataFrame
df = dataset.to_pandas_dataframe()


def plot_data(df, target):
    df = df.dropna()
    X = df.drop(target, axis=1)
    y = df[target]
    t = np.unique(y)
    palette = sns.color_palette(n_colors=len(t))
    clf = LinearDiscriminantAnalysis()
    clf.fit(X, y)
    X_t = clf.transform(X)
    fig = plt.figure()
    ax = plt.subplot(111)
    sns.scatterplot(X_t[:,0], X_t[:,1], hue=y, style=y, palette=palette, ax=ax)
    
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.show()
    run.log_image("embedding", plot=fig)


# Split in train and val
# ---------------------------------------
# plot_data(df, 'Survived')

y = df.pop('Survived')

# Take a hold out set randomly
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=42)

# Create an LGBM dataset for training
categorical_features = ['Alone', 'Sex', 'Pclass', 'Embarked']
train_data = lgbm.Dataset(data=X_train, label=y_train, categorical_feature=categorical_features, free_raw_data=False)

# Create an LGBM dataset from the test
test_data = lgbm.Dataset(data=X_test, label=y_test, categorical_feature=categorical_features, free_raw_data=False)


# Parse parameters
# ---------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument('--boosting', type=str, dest='boosting', default='dart')
parser.add_argument('--num-boost-round', type=int, dest='num_boost_round', default=500)
parser.add_argument('--early-stopping-rounds', type=int, dest='early_stopping_rounds', default=200)
parser.add_argument('--drop-rate', type=float, dest='drop_rate', default=0.15)
parser.add_argument('--learning-rate', type=float, dest='learning_rate', default=0.001)
parser.add_argument('--min-data-in-leaf', type=int, dest='min_data_in_leaf', default=20)
parser.add_argument('--feature-fraction', type=float, dest='feature_fraction', default=0.7)
parser.add_argument('--num-leaves', type=int, dest='num_leaves', default=40)
args = parser.parse_args()

lgbm_params = {
    'application': 'binary',
    'metric': 'binary_logloss',
    'learning_rate': args.learning_rate,
    'boosting': args.boosting,
    'drop_rate': args.drop_rate,
    'min_data_in_leaf': args.min_data_in_leaf,
    'feature_fraction': args.feature_fraction,
    'num_leaves': args.num_leaves,
}


# Logging
# ---------------------------------------
def azure_ml_callback(run):
    def callback(env):
        if env.evaluation_result_list:
            run.log('iteration', env.iteration + 1)
            for data_name, eval_name, result, _ in env.evaluation_result_list:
                run.log("%s (%s)" % (eval_name, data_name), result)
    callback.order = 10
    return callback

def log_importance(clf, run):
    fig, ax = plt.subplots(1, 1)
    lgbm.plot_importance(clf, ax=ax)
    run.log_image("feature importance", plot=fig)

def log_params(params):
    for k, v in params.items():
        run.log(k, v)
    
def log_metrics(clf, X_test, y_test, run):
    preds = np.round(clf.predict(X_test))
    run.log("accuracy (test)", accuracy_score(y_test, preds))
    run.log("precision (test)", precision_score(y_test, preds))
    run.log("recall (test)", recall_score(y_test, preds))
    run.log("f1 (test)", f1_score(y_test, preds))


# Register model
# ---------------------------------------

def log_model(clf):
    joblib.dump(clf, 'outputs/lgbm.pkl')
    run.upload_file('lgbm.pkl', 'outputs/lgbm.pkl')
    run.register_model(model_name='lgbm_titanic', model_path='outputs/lgbm.pkl')


# Train
# ---------------------------------------
evaluation_results = {}
clf = lgbm.train(train_set=train_data,
                 params=lgbm_params,
                 valid_sets=[train_data, test_data], 
                 valid_names=['train', 'val'],
                 evals_result=evaluation_results,
                 num_boost_round=args.num_boost_round,
                 early_stopping_rounds=args.early_stopping_rounds,
                 verbose_eval=20,
                 callbacks = [azure_ml_callback(run)]
                )

log_metrics(clf, X_test, y_test, run)
log_importance(clf, run)
log_model(clf)
log_params(lgbm_params)