#!/usr/bin/env python
# coding: utf-8



from preprocess import Preprocessor

import json
import pandas as pd
import re 
from datetime import datetime
import matplotlib.pyplot as plt



if __name__ == '__main__':
    with open('./data/anonymized_project.json', 'r') as f:
        data = json.load(f)



# loading the dataframe by instantiating the Preprocessor class

d = Preprocessor(data)
df = pd.DataFrame(d.preprocess())


# changing the type of some columns, creating new ones for further analysis

df['created_at'] = pd.to_datetime(df['created_at'])
df.set_index('created_at', inplace=True)

df['image_name'] = df['image_url'].str.extract('\/(\w+)\.jpg$').fillna('')

# dropping the columns where all values are the same - optional

#cols_to_drop = [col for col in df.columns if df[col].nunique() == 1]
#df.drop(cols_to_drop, axis=1, inplace=True)



# loading the ref dataset

with open('./data/references.json', 'r') as f:
    ref = json.load(f)


references = pd.DataFrame([(k, v['is_bicycle']) for k, v in ref.items()], columns=['image_name', 'is_bicycle'])
references.head()


# Analysis
'''
#1. Gather insights about the annotators:
#a. How many annotators did contribute to the dataset?
b. What are the average, min and max annotation times (durations)? Feel free to
add visual representations here such as graphs if you like.
c. Did all annotators produce the same amount of results, or are there
differences?
d. Are there questions for which annotators highly disagree?
2. Besides picking yes or no the annotators had the chance to tell if the data were
corrupted or if they for any reason were not able to solve the task. These are fields
'cant_solve' and 'corrupt_data' given in the task_output.
a. How often does each occur in the project and do you see a trend within the
annotators that made use of these options?

3. Is the reference set balanced? Please demonstrate via numbers and visualizations.
4. Using the reference set, can you identify good and bad annotators? Please use
statistics and visualizations. Feel free to get creative.
'''


# a. How many annotators did contribute to the dataset?

print("{} annotators contributed to the dataset".format(df.vendor_user_id.nunique()))


#b. What are the average, min and max annotation times (durations)? Feel free to
# add visual representations here such as graphs if you like.

df.duration_ms.hist()


# negative duration does not make a lot of sense

df = df[df.duration_ms > 0]

print("Maximum duration: {:.1f} ms\nMinimum duration: {:.1f} ms\nAverage duration: {:.1f} ms".format(df.duration_ms.max(), df.duration_ms.min(), df.duration_ms.mean()))


plt.figure(figsize=(10,6))
ax=df.duration_ms.hist(bins=30)
plt.axvline(df.duration_ms.mean(), color='r', linestyle='--', label='Mean')
plt.axvline(df.duration_ms.max(), color='g', linestyle='--', label='Max')
plt.axvline(df.duration_ms.min(), color='b', linestyle='--', label='Min')
plt.legend(loc='lower right')
plt.show()


# c. Did all annotators produce the same amount of results, or are there differences?

plt.figure(figsize=(15, 8))

counts = df.vendor_user_id.value_counts()
ax = counts.plot(kind='bar')

for i, count in enumerate(counts):
    ax.text(i, count+0.1, str(count), ha='center', va='bottom')
plt.title("Number of annotated images by individual annotator")

plt.show()


# d. Are there questions for which annotators highly disagree?

answ = pd.pivot_table(df, index='image_name', columns='answer', values='vendor_user_id', aggfunc='count').fillna(0)
answ = answ.reset_index()


# calculating the percentage of each answer

answ['yes_pct'] = answ.apply(lambda row: (row['yes'] / (row['yes'] + row['no'])) * 100, axis=1)
answ['no_pct'] = answ.apply(lambda row: (row['no'] / (row['yes'] + row['no'])) * 100, axis=1)


answ['agreement'] = abs(answ.yes_pct - answ.no_pct)
# absolute value difference bewteen the number of responses, the lower the value, the higher the disagreement


answ = answ.sort_values(by='agreement')
answ[answ.agreement == 0]


print("The users have polar opinions on {} pictures".format(len(answ[answ.agreement == 0])))

#2. Besides picking yes or no the annotators had the chance to tell if the data were
# corrupted or if they for any reason were not able to solve the task. These are fields
# 'cant_solve' and 'corrupt_data' given in the task_output.

# a. How often does each occur in the project and do you see a trend within the
# annotators that made use of these options?


print("Marked as corrupt: {} responses, {:.3f}% from total responses\nCouldn't solve: {} responses, {:.3f}% from total responses\n".format(len(df[df.corrupt_data == True]), 
                                                                                                                                          len(df[df.corrupt_data == True]) / len(df) * 100, 
                                                                                                                                          len(df[df.cant_solve == True]), 
                                                                                                                                          len(df[df.cant_solve == True]) / len(df) * 100))
# by individual annotators

print('Cant solve:\n', df[(df.cant_solve == True)].groupby('vendor_user_id').size())
print('Marked data as corrupt:\n',df[(df.corrupt_data == True)].groupby('vendor_user_id').size())


# by anotators: both marked as corrupt and couldn't solve

df[(df.cant_solve == True) | (df.corrupt_data == True)].groupby('vendor_user_id').size().sort_values(ascending=False)

#3. Is the reference set balanced? Please demonstrate via numbers and visualizations.

plt.figure(figsize=(8, 6))

counts = references['is_bicycle'].value_counts()
ax = counts.plot(kind='bar')

for i, count in enumerate(counts):
    ax.text(i, count+0.1, str(count), ha='center', va='bottom')

plt.title("Reference set")
plt.show()

# seems to be quite balanced


df.groupby('answer').size() # as well as the answers

#4. Using the reference set, can you identify good and bad annotators? Please use
# statistics and visualizations. Feel free to get creative.


df = pd.merge(df, references, on='image_name')
df.answer = df.answer.replace({'yes': True, 'no': False}).astype(bool)


# total metrics: with sklearn
from sklearn import metrics

accuracy = metrics.accuracy_score(df.is_bicycle, df.answer)
precision = metrics.precision_score(df.is_bicycle, df.answer)
recall = metrics.recall_score(df.is_bicycle, df.answer)

print("Total accuracy: {:.3f} \nTotal precision: {:.3f} \nTotal recall: {:.3f}".format(accuracy, precision, recall)) 


# or if we want to do it by hand:
tp = sum(df["is_bicycle"] & df["answer"])
fp = sum((df["is_bicycle"] == 0) & (df["answer"] == 1))
tn = sum((df["is_bicycle"] == 0) & (df["answer"] == 0))
fn = sum(df["is_bicycle"] & (df["answer"] == 0))
precision_ = tp / (tp + fp)
recall_ = tp / (tp + fn)
accuracy_ = (tp + tn) / len(df)

print("Total accuracy: {:.3f} \nTotal precision: {:.3f} \nTotal recall: {:.3f}".format(accuracy_, precision_, recall_)) 


# another metric: Cohen's kappa
from sklearn.metrics import cohen_kappa_score

kappa = cohen_kappa_score(df.is_bicycle, df.answer)
print("Cohen's kappa:", kappa)
# kappa is smaller than total accuracy, but that is normal - TBD on the presentation


# by individual annotators
annotators_ranked = df[df['answer'] == df['is_bicycle']].groupby('vendor_user_id').size().reset_index().sort_values(by=0,ascending=False)
annotators_ranked.rename(columns={0: 'correct'}, inplace=True)


total_counts = df['vendor_user_id'].value_counts().to_frame().reset_index().rename(columns={'index': 'vendor_user_id', 'vendor_user_id': 'total_count'})
annotators_ranked = pd.merge(annotators_ranked, total_counts, on='vendor_user_id')


results = []
for user_id, group in df.groupby("vendor_user_id"):
    tp = sum(group["is_bicycle"] & group["answer"])
    fp = sum((group["is_bicycle"] == 0) & (group["answer"] == 1))
    tn = sum((group["is_bicycle"] == 0) & (group["answer"] == 0))
    fn = sum(group["is_bicycle"] & (group["answer"] == 0))
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    accuracy = (tp + tn) / len(group)
    results.append({"vendor_user_id": user_id, "precision": precision, "recall": recall, "accuracy": accuracy})

acc_results = pd.DataFrame(results)


annotators_ranked = pd.merge(annotators_ranked, acc_results, on='vendor_user_id')
annotators_ranked = annotators_ranked.sort_values(by="accuracy", ascending=False)

best_annotator = annotators_ranked.iloc[0]["vendor_user_id"]
worst_annotator = annotators_ranked.iloc[-1]["vendor_user_id"]


# best / worst annotator

print("The user with the total highest accuracy is {} with accuracy {:.3f}".format(best_annotator, annotators_ranked.iloc[0]["accuracy"]))
print("The user with the total lowest accuracy is {} with accuracy {:.3f}".format(worst_annotator, annotators_ranked.iloc[-1]["accuracy"]))

annotators_ranked = annotators_ranked.sort_values(by="total_count", ascending=False)

plt.figure(figsize=(15, 8))

plt.bar(annotators_ranked['vendor_user_id'], annotators_ranked['correct'], label='Correct')
plt.bar(annotators_ranked['vendor_user_id'], annotators_ranked['total_count'] - annotators_ranked['correct'], bottom=annotators_ranked['correct'], label='Incorrect')
plt.legend()
plt.title('Accuracy by Annotator (image count)')
#plt.xlabel('Annotator')
plt.ylabel('Count')
plt.xticks(rotation=90)

plt.show()



# optional: ROC AUC curve (but maybe more of help, when the dataset is imbalanced)

from sklearn.metrics import roc_auc_score

roc_auc_scores = []

for user_id, user_df in df.groupby("vendor_user_id"):
    roc_auc = roc_auc_score(user_df["is_bicycle"], user_df["answer"])
    roc_auc_scores.append((user_id, roc_auc))

roc_auc_df = pd.DataFrame(roc_auc_scores, columns=["user_id", "roc_auc"])
print(roc_auc_df)


from sklearn.metrics import roc_curve, auc

fpr, tpr, thresholds = roc_curve(df["is_bicycle"], df["answer"])
roc_auc = auc(fpr, tpr)



plt.figure(figsize=(15, 8))

plt.plot(fpr, tpr, label=f"AUC={roc_auc:.2f}")
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend()
plt.show()

