import pandas as pd

GROUPS = ['NT', 'TS', 'HC']

def gen_matching_lists():
    nt_df = pd.read_csv('nt_result.csv', index_col=0)
    r01_df = pd.read_csv('r01_result.csv', index_col=0)

    df = pd.concat([nt_df, r01_df])[['sex_screen', 'age_screen', 'group_screen', 'adhd_ever_screen', 'expert_diagnosis_adhd_screen']]
    df = df[df['age_screen'] < 10]
    df.rename(columns={'adhd_ever_screen': 'ksads_adhd_screen'}, inplace=True)

    groups = [ group for group in df['group_screen'].unique().tolist() if group in GROUPS ]
    for group in groups:
        group_df = df[df['group_screen'] == group].sort_values(['sex_screen', 'age_screen'])
        group_df.drop('group_screen', axis=1, inplace=True)
        group_df.to_csv('_'.join([group, 'matching.csv']))

    
gen_matching_lists()
