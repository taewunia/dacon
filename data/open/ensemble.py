import pandas as pd

df1 = pd.read_csv('/Users/choetaewon/Documents/GitHub/bacon/data/open/submission_res.csv')
df2 = pd.read_csv('/Users/choetaewon/Documents/GitHub/bacon/submission_trans.csv')
df3 = pd.read_csv('/Users/choetaewon/Documents/GitHub/bacon/data/open/submission_convnextv2_base.csv')

df_ensemble = df1.copy()


df_ensemble['stable_prob'] = (df1['stable_prob'] + df2['stable_prob'] + df3['stable_prob']) / 3
df_ensemble['unstable_prob'] = (df1['unstable_prob'] + df2['unstable_prob'] + df3['unstable_prob']) / 3


df_ensemble.to_csv('final3_ensemble.csv', index=False)

print("완료!!!")