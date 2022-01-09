from sklearn.model_selection import GroupShuffleSplit
import pandas as pd

df = pd.read_csv('cleaning_dataset/pt4/Detail_Incident_Activity_cut6_pt4.csv', sep=',', low_memory=False)

splitter = GroupShuffleSplit(test_size=.20, n_splits=2, random_state = 7)
split = splitter.split(df, groups=df['incident_id'])
train_inds, test_inds = next(split)

train = df.iloc[train_inds]
test = df.iloc[test_inds]

train.to_csv('cleaning_dataset/train_test/TRAIN_PREPROCESSED_Detail_Incident_Activity_cut6.csv', sep=';', encoding='utf-8', index=False)
test.to_csv('cleaning_dataset/train_test/TEST_PREPROCESSED_Detail_Incident_Activity_cut6.csv', sep=';', encoding='utf-8', index=False)
