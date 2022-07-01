import pickle
import pandas
import pandas as pd
from sklearn.preprocessing import LabelEncoder
model = ['RF.sav', 'svc.sav', 'svcl.sav', 'svcp.sav', 'svcr.sav', 'svcs.sav', 'GNB.sav', 'DT.sav', 'LR.sav']


new_song=[{'acousticness':0.0301,
          'danceability':0.583,
           'duration_ms':224092,
           'energy':0.891,
           'instrumentalness':0.000003,
           'key':7,
           'liveness':0.129,
           'loudness':-3.495,
           'mode':1,
           'speechiness':0.447,
           'tempo':149.843,
           'time_signature':4.0,
           'valence':0.321,
           'song_title':'Without U',
           'artist':'Steve Aoki'}]
df = pd.DataFrame(new_song)
encoder = LabelEncoder()
df['song_title'] = encoder.fit_transform(df['song_title'])
df['artist'] = encoder.fit_transform(df['artist'])
print(df)
res = {1: 'yes',
       0: 'no'}
for i in range(len(model)):
    loaded_model = pickle.load(open(model[i], 'rb'))
    result = loaded_model.predict(df)
    print(f'{model[i]} = {res.get(result[0])}')


