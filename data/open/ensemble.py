import pandas as pd

# 🔥 1. 두 모델이 푼 답안지(CSV) 불러오기
df1 = pd.read_csv('/Users/choetaewon/Documents/GitHub/bacon/data/open/submission_res.csv')
df2 = pd.read_csv('/Users/choetaewon/Documents/GitHub/bacon/submission_trans.csv')


df_ensemble = df1.copy()


df_ensemble['stable_prob'] = (df1['stable_prob'] + df2['stable_prob']) / 2
df_ensemble['unstable_prob'] = (df1['unstable_prob'] + df2['unstable_prob']) / 2


df_ensemble.to_csv('final2_ensemble.csv', index=False)

print("완료!!!")