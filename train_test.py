from sklearn.model_selection import GroupShuffleSplit
import pandas as pd

df = pd.read_csv('cleaning_dataset/pt4/Detail_Incident_Activity_cut6_pt4.csv', sep=',', low_memory=False)

df.drop(columns=["Unnamed: 0", "r"],inplace=True)

df['sojourn_set_stc'] = 0
df['sojourn_mset_stc'] = 0
df['sojourn_seq_stc'] = 0
df['seq_state_id'] = 0
df['set_state_id'] = 0
df['mset_state_id'] = 0

splitter = GroupShuffleSplit(test_size=.40, n_splits=2, random_state = 7)
split = splitter.split(df, groups=df['incident_id'])
train_inds, test_inds = next(split)


#rename column for ATS 
df_ATS = df
df_ATS = df_ATS.rename(columns={'incident_id': 'number', 'datestamp' : 'updated_at', 'incidentactivity_type' : 'incident_state'})

train = df_ATS.iloc[train_inds]
test = df_ATS.iloc[test_inds]

train.to_csv('cleaning_dataset/train_test/ATS_1_6040_TRAIN_PREPROCESSED_Detail_Incident_Activity_cut6.csv', sep=',', encoding='utf-8', index=False)
test.to_csv('cleaning_dataset/train_test/ATS_1_6040_TEST_PREPROCESSED_Detail_Incident_Activity_cut6.csv', sep=',', encoding='utf-8', index=False)

#rename column for LSTM
# df_LSTM = df
# df_LSTM = df_LSTM.rename(columns={'incident_id': 'CaseID', 'datestamp' : 'CompleteTimestamp', 'incidentactivity_type' : 'Activity'})
# df_LSTM.Activity = pd.Categorical(df_LSTM.Activity)
# df_LSTM['ActivityID'] = df_LSTM.Activity.cat.codes
# df_LSTM = df_LSTM[['CaseID', 'ActivityID', 'CompleteTimestamp']]
# 
# train = df_LSTM.iloc[train_inds]
# test = df_LSTM.iloc[test_inds]
# 
# train.to_csv('cleaning_dataset/train_test/LSTM_1_TRAIN_PREPROCESSED_Detail_Incident_Activity_cut6.csv', sep=',', encoding='utf-8', index=False)
# test.to_csv('cleaning_dataset/train_test/LSTM_1_TEST_PREPROCESSED_Detail_Incident_Activity_cut6.csv', sep=',', encoding='utf-8', index=False)
