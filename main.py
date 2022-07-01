import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import seaborn as sns
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import pickle

# Importing the spotify the data set
spotify_df = pd.read_csv("data.csv")
print("Dataset\n", spotify_df.describe())

# Removing outliers using Z-score
def remove_outliers_gaussian(df, name='feature'):
    mean, std = np.mean(df[name]), np.std(df[name])
    std_cut = 3 * std
    lower, upper = mean - std_cut, mean + std_cut
    df = df[(df[name] >= lower) & (df[name] <= upper)]
    return df

spotify_df1 = spotify_df.copy()
spotify_df1 = remove_outliers_gaussian(spotify_df1, 'acousticness')  # Removing the outliers of acousticness feature.
print("rows in the dataset:\n", len(spotify_df1))

# Creating training, validation and testing set using train_test_split.
train_val_df, test_df = train_test_split(spotify_df1, test_size=0.2, random_state=2)
train_df, val_df = train_test_split(train_val_df, test_size=0.25, random_state=2)
print('\nTraining data size:', train_df.shape)
print('\nValidation data size:', val_df.shape)
print('\nTesting data size', test_df.shape)

inputs_cols = spotify_df.columns.tolist()[:-1]  # includes all the features except target as input
target_col = "target"  # Output

print('\ninputs: {}'.format(inputs_cols))
print('\ntarget: {}'.format(target_col))

# Create train inputs and target
train_inputs = train_df[inputs_cols].copy()
train_target = train_df[target_col].copy()

# Create val inputs and target
val_inputs = val_df[inputs_cols].copy()
val_target = val_df[target_col].copy()

# Create test inputs and target
test_inputs = test_df[inputs_cols].copy()
test_target = test_df[target_col].copy()

numerical_cols = train_inputs.select_dtypes(include=['float64', 'int64']).columns.tolist()
print("\nNumerical columns:", numerical_cols)
encoded_cols = ['mode']
categorical_cols = train_inputs.select_dtypes(include=['object']).columns.tolist()
print("\nCategorical columns:", categorical_cols)
scaler = MinMaxScaler()

train_inputs[numerical_cols] = scaler.fit_transform(train_inputs[numerical_cols])
val_inputs[numerical_cols] = scaler.fit_transform(val_inputs[numerical_cols])
test_inputs[numerical_cols] = scaler.fit_transform(test_inputs[numerical_cols])

print("\nFeatures after scaling:\n", test_inputs[numerical_cols])

encoder = LabelEncoder()
train_inputs['song_title'] = encoder.fit_transform(train_inputs['song_title'])
val_inputs['song_title'] = encoder.fit_transform(val_inputs['song_title'])
test_inputs['song_title'] = encoder.fit_transform(test_inputs['song_title'])
# train_inputs['artist'] = encoder.fit_transform(train_inputs['artist'])
# val_inputs['artist'] = encoder.fit_transform(val_inputs['artist'])
# test_inputs['artist'] = encoder.fit_transform(test_inputs['artist'])

encoded_cols = ['song_title']

print("\nFeatures after label encoding:\n", train_inputs[categorical_cols])

X_train = train_inputs[numerical_cols + encoded_cols]
X_val = val_inputs[numerical_cols + encoded_cols]
X_test = test_inputs[numerical_cols + encoded_cols]

# ***********Logistic Regression**************************************************************
logistic_reg = LogisticRegression(solver='liblinear', max_iter=1000).fit(X_train, train_target)
train_preds = logistic_reg.predict(X_train)
logistic_train_acc = accuracy_score(train_target, train_preds)
print('\nLogistic regression Train Accuracy:{:.2f}%'.format(logistic_train_acc * 100))
val_preds = logistic_reg.predict(X_val)
logistic_val_acc = accuracy_score(val_target, val_preds)
print('\nLogistic regression Validation Accuracy:{:.2f}%'.format(logistic_val_acc * 100))
test_preds = logistic_reg.predict(X_test)
logistic_test_acc = accuracy_score(test_target, test_preds)
print('\nLogistic regression Testing Accuracy:{:.2f}%'.format(logistic_test_acc * 100))
pickle.dump(logistic_reg, open('LR.sav', 'wb'))
# ************************SVM***************************************
svc = SVC()

model = svc.fit(X_train, train_target)

train_preds = model.predict(X_train)
svm_train_acc = accuracy_score(train_target, train_preds)

val_preds = model.predict(X_val)
svm_val_acc = accuracy_score(val_target, val_preds)

test_preds = model.predict(X_test)
svm_test_acc = accuracy_score(test_target, test_preds)

print('\nTraining accuracy score with default hyperparameters: {:0.2f}%'.format(svm_train_acc * 100))
print('\nValidation accuracy score with default hyperparameters: {:0.2f}%'.format(svm_val_acc * 100))
print('\nTesting accuracy score with default hyperparameters: {:0.2f}%'.format(svm_test_acc * 100))

pickle.dump(model, open('svc.sav', 'wb'))

poly_svc = SVC(kernel='poly', C=100.0)

model = poly_svc.fit(X_train, train_target)

train_preds = model.predict(X_train)
svm_train_acc = accuracy_score(train_target, train_preds)

val_preds = model.predict(X_val)
svm_val_acc = accuracy_score(val_target, val_preds)

test_preds = model.predict(X_test)
svm_test_acc = accuracy_score(test_target, test_preds)

print('\nTraining accuracy score with polynomial hyperparameters: {:0.2f}%'.format(svm_train_acc * 100))
print('\nValidation accuracy score with polynomial hyperparameters: {:0.2f}%'.format(svm_val_acc * 100))
print('\nTesting accuracy score with polynomial hyperparameters: {:0.2f}%'.format(svm_test_acc * 100))

pickle.dump(model, open('svcp.sav', 'wb'))

linear_svc = SVC(kernel='linear', C=100.0)

model = linear_svc.fit(X_train, train_target)

train_preds = model.predict(X_train)
svm_train_acc = accuracy_score(train_target, train_preds)

val_preds = model.predict(X_val)
svm_val_acc = accuracy_score(val_target, val_preds)

test_preds = model.predict(X_test)
svm_test_acc = accuracy_score(test_target, test_preds)

print('\nTraining accuracy score with linear hyperparameters: {:0.2f}%'.format(svm_train_acc * 100))
print('\nValidation accuracy score with linear hyperparameters: {:0.2f}%'.format(svm_val_acc * 100))
print('\nTesting accuracy score with linear hyperparameters: {:0.2f}%'.format(svm_test_acc * 100))

pickle.dump(model, open('svcl.sav', 'wb'))

sigmoid_svc = SVC(kernel='sigmoid', C=100.0)

model = sigmoid_svc.fit(X_train, train_target)

train_preds = model.predict(X_train)
svm_train_acc = accuracy_score(train_target, train_preds)

val_preds = model.predict(X_val)
svm_val_acc = accuracy_score(val_target, val_preds)

test_preds = model.predict(X_test)
svm_test_acc = accuracy_score(test_target, test_preds)

print('\nTraining accuracy score with sigmoid hyperparameters: {:0.2f}%'.format(svm_train_acc * 100))
print('\nValidation accuracy score with sigmoid hyperparameters: {:0.2f}%'.format(svm_val_acc * 100))
print('\nTesting accuracy score with sigmoid hyperparameters: {:0.2f}%'.format(svm_test_acc * 100))

pickle.dump(model, open('svcs.sav', 'wb'))

RBF_svc = SVC(kernel='rbf', C=100.0)

model = RBF_svc.fit(X_train, train_target)

train_preds = model.predict(X_train)
svm_train_acc = accuracy_score(train_target, train_preds)

val_preds = model.predict(X_val)
svm_val_acc = accuracy_score(val_target, val_preds)

test_preds = model.predict(X_test)
svm_test_acc = accuracy_score(test_target, test_preds)

print('\nTraining accuracy score with rbf hyperparameters: {:0.2f}%'.format(svm_train_acc * 100))
print('\nValidation accuracy score with rbf hyperparameters: {:0.2f}%'.format(svm_val_acc * 100))
print('\nTesting accuracy score with rbf hyperparameters: {:0.2f}%'.format(svm_test_acc * 100))

pickle.dump(model, open('svcr.sav', 'wb'))

# *********************NaiveBayes******************************************************************

gnb = GaussianNB()

model = gnb.fit(X_train, train_target)

train_preds = model.predict(X_train)
NB_train_acc = accuracy_score(train_target, train_preds)

val_preds = model.predict(X_val)
NB_val_acc = accuracy_score(val_target, val_preds)

test_preds = model.predict(X_test)
NB_test_acc = accuracy_score(test_target, test_preds)

print('\nTraining accuracy score with NB: {:0.2f}%'.format(NB_train_acc * 100))
print('\nValidation accuracy score with NB: {:0.2f}%'.format(NB_val_acc * 100))
print('\nTesting accuracy score with NB: {:0.2f}%'.format(NB_test_acc * 100))

pickle.dump(model, open('GNB.sav', 'wb'))

# **************************Decision Tree******************************************************************
spotify_df2 = spotify_df.copy()

spotify_df2 = remove_outliers_gaussian(spotify_df2, 'danceability')
print('\nSize of dataset: ', len(spotify_df2))

model = DecisionTreeClassifier(random_state=2).fit(X_train, train_target)

train_preds = model.predict(X_train)

DT_train_acc = accuracy_score(train_target, train_preds)
print('\nDecision Tree Train Accuracy:{:.2f}%'.format(DT_train_acc * 100))

val_preds = model.predict(X_val)
DT_val_acc = accuracy_score(val_target, val_preds)
print('\nValidation Tree Accuracy:{:.2f}%'.format(DT_val_acc * 100))

test_preds = model.predict(X_test)
DT_test_acc = accuracy_score(test_target, test_preds)

from sklearn.tree import plot_tree, export_text

plt.figure(figsize=(15, 15))
plot_tree(model, feature_names=X_train.columns, max_depth=3, filled=True)
# plt.show()

print("\nMax depth of the decision tree: ", model.tree_.max_depth)

tree_text = export_text(model, max_depth=10, feature_names=list(X_train.columns))
print(tree_text[:5000])

print(model.feature_importances_)

importance_df = pd.DataFrame({
    'feature': X_train.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(importance_df.head(10))

plt.title('Feature Importance')
sns.barplot(data=importance_df.head(10), x='importance', y='feature')
# plt.show()

def max_depth_error(md):
    model = DecisionTreeClassifier(max_depth=md, random_state=2)
    model.fit(X_train, train_target)
    train_acc = 1 - model.score(X_train, train_target)
    val_acc = 1 - model.score(X_val, val_target)
    test_acc= 1 - model.score(X_test, test_target)

    return {'Max Depth': md, 'Training Error': train_acc, 'Validation Error': val_acc}

model = DecisionTreeClassifier(max_depth=9, random_state=2).fit(X_train, train_target)

print('\nTraining accuracy of decision tree (2nd turn): {:0.2f}%'.format(model.score(X_train, train_target) * 100))
print("\nValidation accuracy of decision tree (2nd turn): {:0.2f}%".format(model.score(X_val, val_target) * 100))

model = DecisionTreeClassifier(max_depth=7, random_state=2).fit(X_train, train_target)

print('\nTraining accuracy of decision tree (3rd turn): {:0.2f}%'.format(model.score(X_train, train_target) * 100))
print("\nValidation accuracy of decision tree (3rd turn): {:0.2f}%".format(model.score(X_val, val_target) * 100))
print("\nTesting accuracy of decision tree (3rd turn): {:0.2f}%".format(model.score(X_test, test_target) * 100))

pickle.dump(model, open('DT.sav', 'wb'))

# *************************************Random Forest*************************************************************
model = RandomForestClassifier(n_jobs=-1, random_state=2).fit(X_train, train_target)
print(model.score(X_train, train_target))
print(model.score(X_val, val_target))
importance_df = pd.DataFrame({
    'feature': X_train.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
importance_df.head(10)
plt.title('Feature Importance')
sns.barplot(data=importance_df.head(10), x='importance', y='feature')

def n_estimator_error(n):
    model = RandomForestClassifier(n_jobs=-1, n_estimators=n, random_state=2)
    model.fit(X_train, train_target)
    train_acc = 1 - model.score(X_train, train_target)
    val_acc = 1 - model.score(X_val, val_target)
    # test_acc + 1 - model.score(X_train, test_target)

    return {'N_Estimator': n, 'Training Error': train_acc, 'Validation Error': val_acc}

n_errors = pd.DataFrame([n_estimator_error(n) for n in range(100, 500)])
n_errors.sort_values('Validation Error', ascending=True)
plt.figure()
plt.plot(n_errors['N_Estimator'], n_errors['Training Error'])
plt.plot(n_errors['N_Estimator'], n_errors['Validation Error'])
plt.title('Training vs. Validation Error')

plt.xlabel('Number Estimator')
plt.ylabel('Prediction Error (1-Accuracy)')
plt.legend(['Training', 'Validation'])
# plt.show()
model = RandomForestClassifier(n_jobs=-1, n_estimators=112, random_state=2).fit(X_train, train_target)
train_acc = model.score(X_train, train_target)
val_acc = model.score(X_val, val_target)
print('Train_accuracy:{:.2f}\nVal_accuracy:{:.2f}'.format(train_acc * 100, val_acc * 100))

def max_features(n):
    model = RandomForestClassifier(n_jobs=-1, max_features=n, random_state=2)
    model.fit(X_train, train_target)
    train_acc = 1 - model.score(X_train, train_target)
    val_acc = 1 - model.score(X_val, val_target)

    return {'Max Feature': n, 'Training Error': train_acc, 'Validation Error': val_acc}

feature_error = pd.DataFrame(max_features(n) for n in range(1, 12))
feature_error.sort_values('Validation Error', ascending=True)
plt.figure()
plt.plot(feature_error['Max Feature'], feature_error['Training Error'])
plt.plot(feature_error['Max Feature'], feature_error['Validation Error'])
plt.title('Training vs. Validation Error')
plt.xlabel('Max Feature')
plt.ylabel('Prediction Error (1-Accuracy)')
plt.legend(['Training', 'Validation'])
# plt.show()
model = RandomForestClassifier(n_jobs=-1, max_features=3, random_state=2).fit(X_train, train_target)
train_acc = model.score(X_train, train_target)
val_acc = model.score(X_val, val_target)
preds = model.predict(X_val)

print('Train_accuracy:{:.2f}\nVal_accuracy:{:.2f}'.format(train_acc * 100, val_acc * 100))

def max_depth_error(n):
    model = RandomForestClassifier(n_jobs=-1, max_depth=n, random_state=2).fit(X_train, train_target)
    train_acc = 1 - model.score(X_train, train_target)
    val_acc = 1 - model.score(X_val, val_target)
    return {'Max Depth': n, 'Training Error': train_acc, 'Validation Error': val_acc}

md_error = pd.DataFrame([max_depth_error(n) for n in range(1, 25)])
md_error.sort_values('Validation Error', ascending=True)
plt.figure()
plt.plot(md_error['Max Depth'], md_error['Training Error'])
plt.plot(md_error['Max Depth'], md_error['Validation Error'])
plt.title('Training vs. Validation Error')
plt.xlabel('Max. Depth')
plt.ylabel('Prediction Error (1-Accuracy)')
plt.legend(['Training', 'Validation'])
## plt.show()
model = RandomForestClassifier(n_jobs=-1, n_estimators=112, max_features=3, max_depth=16, random_state=2).fit(X_train,
                                                                                                              train_target)
model.score(X_val, val_target)
model.score(X_test, test_target)

pickle.dump(model, open('RF.sav', 'wb'))

# new_song={'acousticness':0.0301,
#           'danceability':0.583,
#            'duration_ms':224092,
#            'energy':0.891,
#            'instrumentalness':0.000003,
#            'key':7,
#            'liveness':0.129,
#            'loudness':-3.495,
#            'mode':1,
#            'speechiness':0.447,
#            'tempo':149.843,
#            'time_signature':4.0,
#            'valence':0.321,
#            'song_title':'Without U',
#            'artist':'Steve Aoki'}
#
# def predict_song(new_song):
#     song_df=pd.DataFrame([new_song])
#     X_song=song_df[numerical_cols + encoded_cols]
#     pred=model.predict(X_song)[0]
#     prob=model.predict_proba(X_song)[0][list(model.classes_).index(pred)]
#     if pred == 0:
#         return 'Discard', prob
#     elif pred == 1:
#         return 'Save', prob
# print(predict_song(new_song))