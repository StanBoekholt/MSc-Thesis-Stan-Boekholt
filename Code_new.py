# Load libraries
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import ClusterCentroids
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_excel('//studfiles.campus.uvt.nl/files/home/home00/u153190/MSc Data Science/Msc Thesis/Thesis/Thesis - Python/Dataset Thesis cleaned.xlsx')

# Checking the data for missing values
df.isnull().sum()

# Define the target variable
target_variable = df['turnover']

# Define predictor variables 
variables = df[['corruption', 'patronage', 'widespread', 'abuse', 'bidding', 'procurement_national', 
                'procurement_local', 'transparency', 'connections', 'favouritism', 'petty_corruption',
                'bribing', 'reported', 'court', 'fined', 'sector', 'employees', 'age']]

x = variables
y = target_variable

# Split the dataset into training set (70%) and testing set (30%)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1, stratify=y)

# 5-fold cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

# Baseline Decision Tree
baseline_dt = DecisionTreeClassifier(random_state=1)
baseline_dt.fit(x_train, y_train)
baseline_dt_predictions = baseline_dt.predict(x_test)

print("Baseline Decision Tree Precision:", precision_score(y_test, baseline_dt_predictions, average='weighted'))
print("Baseline Decision Tree Recall:", recall_score(y_test, baseline_dt_predictions, average='weighted'))
print("Baseline Decision Tree F1 Score:", f1_score(y_test, baseline_dt_predictions, average='weighted'))

# Baseline Random Forest
baseline_rf = RandomForestClassifier(random_state=1)
baseline_rf.fit(x_train, y_train)
baseline_rf_predictions = baseline_rf.predict(x_test)

print("Baseline Random Forest Precision:", precision_score(y_test, baseline_rf_predictions, average='weighted'))
print("Baseline Random Forest Recall:", recall_score(y_test, baseline_rf_predictions, average='weighted'))
print("Baseline Random Forest F1 Score:", f1_score(y_test, baseline_rf_predictions, average='weighted'))

# Test values for hyperparameters Decision Tree, only for testing.
hyper_para_dt = {'max_depth': [None] + list(range(1, 21)), 'min_samples_split': [None] + list(range(1, 21))}
# Test values for hyperparameters Random Forest, only for testing
hyper_para_rf = {'max_depth': [None] + list(range(1, 21)), 'n_estimators': [50, 100, 150, 200, 250]}

# Apply SMOTE
smote = SMOTE(sampling_strategy='auto', random_state=1)
x_train_smote, y_train_smote = smote.fit_resample(x_train, y_train)

# Apply optimal hyperparameters DT with SMOTE
hyper_para_dt = {'max_depth': [5], 'min_samples_split': [14]}

# Decision Tree with SMOTE and tuned hyperparameters
smote_dt = DecisionTreeClassifier(random_state=1)
random_search_smote_dt = RandomizedSearchCV(smote_dt, param_distributions=hyper_para_dt)
random_search_smote_dt.fit(x_train_smote, y_train_smote)

best_smote_dt = random_search_smote_dt.best_estimator_
best_smote_dt_predictions = best_smote_dt.predict(x_test)

print("SMOTE Decision Tree Precision:", precision_score(y_test, best_smote_dt_predictions, average='weighted'))
print("SMOTE Decision Tree Recall:", recall_score(y_test, best_smote_dt_predictions, average='weighted'))
print("SMOTE Decision Tree F1 Score:", f1_score(y_test, best_smote_dt_predictions, average='weighted'))

print("Optimal hyperparameters Decision Tree (SMOTE):", random_search_smote_dt.best_params_)

# Apply optimal hyperparameters RF with SMOTE
hyper_para_rf = {'max_depth': [19], 'n_estimators': [250]}

# Random Forest with SMOTE and tuned hyperparameters
smote_rf = RandomForestClassifier(random_state=1)
random_search_smote_rf = RandomizedSearchCV(smote_rf, param_distributions=hyper_para_rf)
random_search_smote_rf.fit(x_train_smote, y_train_smote)

best_smote_rf = random_search_smote_rf.best_estimator_
best_smote_rf_predictions = best_smote_rf.predict(x_test)

print("SMOTE Random Forest Precision:", precision_score(y_test, best_smote_rf_predictions, average='weighted'))
print("SMOTE Random Forest Recall:", recall_score(y_test, best_smote_rf_predictions, average='weighted'))
print("SMOTE Random Forest F1 Score:", f1_score(y_test, best_smote_rf_predictions, average='weighted'))

print("Optimal hyperparameters Random Forest (SMOTE):", random_search_smote_rf.best_params_)

# Apply Cluster-based under-sampling
cc = ClusterCentroids(sampling_strategy='auto', random_state=1)
x_train_cc, y_train_cc = cc.fit_resample(x_train, y_train)

# Apply optimal hyperparameters DT with CC
hyper_para_dt = {'max_depth': [2], 'min_samples_split': [1]}

# Decision Tree with Cluster-based under-sampling and tuned hyperparameters
cc_dt = DecisionTreeClassifier(random_state=1)
random_search_cc_dt = RandomizedSearchCV(cc_dt, param_distributions=hyper_para_dt)
random_search_cc_dt.fit(x_train_cc, y_train_cc)

best_cc_dt = random_search_cc_dt.best_estimator_
best_cc_dt_predictions = best_cc_dt.predict(x_test)

print("Cluster-based under-sampling Decision Tree Precision:", precision_score(y_test, best_cc_dt_predictions, average='weighted'))
print("Cluster-based under-sampling Decision Tree Recall:", recall_score(y_test, best_cc_dt_predictions, average='weighted'))
print("Cluster-based under-sampling Decision Tree F1 Score:", f1_score(y_test, best_cc_dt_predictions, average='weighted'))

print("Optimal hyperparameters Decision Tree (Cluster-based under-sampling):", random_search_cc_dt.best_params_)

# Apply optimal hyperparameters RF with SMOTE
hyper_para_rf = {'max_depth': [4], 'n_estimators': [200]}

# Random Forest with cluster-based under-sampling and tuned hyperparameters
cc_rf = RandomForestClassifier(random_state=1)
random_search_cc_rf = RandomizedSearchCV(cc_rf, param_distributions=hyper_para_rf)
random_search_cc_rf.fit(x_train_cc, y_train_cc)

best_cc_rf = random_search_cc_rf.best_estimator_
best_cc_rf_predictions = best_cc_rf.predict(x_test)

print("Cluster-based under-sampling Random Forest Precision:", precision_score(y_test, best_cc_rf_predictions, average='weighted'))
print("Cluster-based under-sampling Random Forest Recall:", recall_score(y_test, best_cc_rf_predictions, average='weighted'))
print("Cluster-based under-sampling Random Forest F1 Score:", f1_score(y_test, best_cc_rf_predictions, average='weighted'))

print("Optimal hyperparameters Random Forest (Cluster-based under-sampling):", random_search_cc_rf.best_params_)

# Check the cross-validation scores for best model (Random Forest SMOTE)
cv_scores = cross_val_score(best_smote_rf, x_train_smote, y_train_smote, cv=5)
print("Cross-Validation Scores:", cv_scores)
print("Mean Cross-Validation Score:", cv_scores.mean())

# Confusion matrix for best model (Random Forest SMOTE)
cm_smote_rf = confusion_matrix(y_test, best_smote_rf_predictions, labels=[1, 2, 3])

# Visualise confusion matrix for best model (Random Forest SMOTE)
labels = ["Increase", "Decrease", "Remained Unchanged"]
sns.heatmap(cm_smote_rf, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, cbar=False)
plt.xlabel('Predicted', weight = 'bold')
plt.ylabel('Actual', weight = 'bold')
plt.show()

# Feature importance
feature_importance = best_smote_rf.feature_importances_

# Create a dataframe to display feature importances
feature_importance_df = pd.DataFrame({'Feature': x_train.columns, 'Importance': feature_importance})

# Sort the dataframe by importance
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Print the feature importances
print("Feature importance:")
print(feature_importance_df)