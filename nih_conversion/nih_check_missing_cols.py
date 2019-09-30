import pandas as pd

redcap_form = 'medical_and_surgical_history'
nih_form = 'mvhsp01'

df = pd.read_csv(r'C:\Users\acevedoh\Documents\NewTics local stuff\nt r01\NewTicsR01_DataDictionary_2018-06-26.csv')
rc_cols = df[df['Form Name'] == redcap_form].iloc[:,0].values

nih_df = pd.read_csv(r'C:\Users\acevedoh\Box\Black_Lab\projects\TS\New Tics R01\Data\NIH Data Archive\conversion\nih_dd\{}_definitions.csv'.format(nih_form))
nih_cols = nih_df['Aliases'].str.extract(',*(\w*)_12mo,*').dropna().values.flatten()

desc = df[df['Field Type'] == 'descriptive'].iloc[:,0].unique()
print([ col for col in rc_cols if col not in nih_cols and col not in desc ])
