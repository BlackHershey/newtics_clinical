import argparse
import json
import numpy as np
import pandas as pd

from extract_cpt import extract_cpt
from score_drz import score_drz
from score_redcap_data import score_redcap_data
from score_weather import score_weather

from getpass import getuser
from os.path import join

RESULT_OUTPUT_FILE = 'all_data_result.csv'
MISSING_FILE_TEMPLATE = '{}_all_data_missing{}.{}'

def concat_df(study_df, other_df):
    return pd.concat([other_df, study_df])

def generate_formatted_table(study_name, nt_file, r01_file, check_missing, use_existing):
    with open('config.json') as config_file:
        config = json.load(config_file)
        study_vars = config[study_name]

    basedir = study_vars['directories']['base'].format(getuser())
    indir = { k: v.format(basedir) for k,v in study_vars['directories']['input'].items() }
    outdir = study_vars['directories']['output'].format(basedir)

    # get dfs for all data sources
    redcap_df = score_redcap_data(study_name, nt_file, r01_file, use_existing=use_existing)
    cpt_df = extract_cpt(indir['CPT'], use_existing=use_existing)
    drz_df = score_drz(indir['DRZ'], use_existing=use_existing)
    weather_df = score_weather(indir['weather'], use_existing=use_existing)

    if study_name == 'r01':
        nt_indir = { k: v.format(config['nt']['directories']['base'].format(getuser())) for k,v in config['nt']['directories']['input'].items() }
        cpt_df = concat_df(cpt_df, extract_cpt(nt_indir['CPT'], use_existing=use_existing))
        drz_df = concat_df(drz_df, score_drz(nt_indir['DRZ'], use_existing=use_existing))
        weather_df = concat_df(weather_df, score_weather(nt_indir['weather'], use_existing=use_existing))

    df = redcap_df.join(cpt_df, how='left')
    df = df.join(drz_df, how='left')
    df = df.join(weather_df, how='left')
    df = df.dropna(axis=1, how='all') # drop columns that are all NaN (i.e. measures that don't apply to certain sessions)

    if check_missing:
        missing_data = df.transpose()
        null_mask = missing_data.isnull()
        missing_data[~null_mask] = np.nan
        missing_data[null_mask] = 'X'
        known_missing  = pd.read_excel(MISSING_FILE_TEMPLATE.format(study_name, '_ANNOTATED', 'xlsm'), index_col=1, skipfooter=4)
        missing_data[known_missing.isin([0,1])] = np.nan
        missing_data = missing_data.dropna(axis=0, how='all')
        missing_data.to_csv(MISSING_FILE_TEMPLATE.format(study_name, '', 'xlsx'))

    df.to_csv(join(outdir, '_'.join([study_name, RESULT_OUTPUT_FILE])))
    return df


def generate_demographic_summary(study_name, df = None, check_missing=False):
    with open('config.json') as config_file:
        study_vars = json.load(config_file)[study_name]

    if df is None:
        df = pd.read_csv('_'.join([study_name, RESULT_OUTPUT_FILE])).set_index('demo_study_id')

    basedir = study_vars['directories']['base'].format(getuser())
    writer = pd.ExcelWriter(join(study_vars['directories']['output'].format(basedir), '{}_summary.xlsx'.format(study_name)))

    columns = [ 'sex_screen', 'non_white_screen', 'age_screen', '12mo_complete', 'kbit_iq_screen', 'days_since_onset', 'ygtss_past_week_expert_total_tic',
        'ygtss_past_week_expert_m_total', 'ygtss_past_week_expert_p_total', 'ygtss_past_week_expert_total_impairment', 'ygtss_minimal_impairment', 'puts_total', 'puts_total_completed_only',
        'dci_total', 'dci_attempts_to_suppress_tics_ever', 'expert_diagnosis_awareness_ever', 'cybocs_past_week_expert_total', 'adhd_current_expert_total', 'srs_tscore_screen', 'ses_avg_screen', 'vocal_tics_ever',
        'complex_tics_ever', 'adhd_ever_screen', 'ocd_ever_screen', 'other_anxiety_disorder_ever_screen', 'other_ksads_ever_screen', 'non_tic_ksads_ever_screen',
        '1st_relative_with_tics', 'dominant_hand_screen', 'peg_dominant_30s_screen', 'omissions_pct', 'commissions_pct', 'hit_rt_n', 'avg_tics_per_minute_baseline',
        'avg_tic_free_10s_per_minute_baseline', 'weather_all']

    columns_12mo_only = ['ygtss_total_tic_delta'] + [ '_'.join(['expert_diagnosis', dx, '12mo']) for dx in ['tourette', 'chronic_tics', 'transient'] ] \
        + [ col for col in df.columns if col.startswith('outcome') and col.endswith('12mo')]

    visits = [ 'screen', '12mo']
    for visit in visits:
        if visit == '12mo':
            df = df[df['12mo_complete'] == True]
            columns += columns_12mo_only
            columns.remove('12mo_complete')

        rows = []
        for col in columns:
            col = col if col in df.columns else '_'.join([col, visit])
            if col not in df.columns:
                print('Missing column:', col)
                continue
            if df[col].isnull().all():
                print('No data for measure:', col)
                continue

            if np.issubdtype(df[col].dtype, np.number):
                mean = df[col].mean(axis=0)
                std = df[col].std(axis=0)
                range_str = '-'.join(['{}'.format(int(num)) if int(num) == num else '{:.2f}'.format(num) for num in [df[col].min(), df[col].max()] ])
                row = [col, '{:.2f} ({:.2f}), {}'.format(mean, std, range_str), df[col].count()]
            else:
                row = [ col ]
                counts = df[col].value_counts().to_dict()
                n = df[col].count()

                keys = list(counts.keys())
                if True in keys or 'Y' in keys:
                    pct_count = counts[True] if True in keys else counts['Y']
                    pct = 100 * (float(pct_count) / n)
                    row.append('{} ({:.2f}%)'.format(pct_count, pct))
                else:
                    count_str = '{} {}'.format(counts[keys[0]], keys[0])
                    if len(keys) > 1:
                        count_str += ' / {} {}'.format(counts[keys[1]], keys[1])
                    row.append(count_str)
                row.append(n)
            rows.append(row)

        summary_columns = ['Descriptor', 'Summary', 'N']
        summary_df = pd.DataFrame(data = rows, columns = summary_columns)
        print(summary_df)
        summary_df.to_excel(writer, sheet_name=visit)
    writer.save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='combine data from all sources (redcap, drz, cpt, weather) to generate summary')
    parser.add_argument('study_name', choices=['nt','r01'], help='which project to summarize data for')
    parser.add_argument('--nt_file', help='file containing data exported from NewTics redcap project (if unspecified API will be used)')
    parser.add_argument('--r01_file', help='file containing data exported from R01 redcap project (if unspecified API will be used)')
    parser.add_argument('--check', action='store_true', help='check for missing/extra data and output anomalies to file (default is not to check)')
    parser.add_argument('--use_existing', action='store_true', help='use existing redcap/drz/cpt/weather output files (default is to recalculate them)')
    args = parser.parse_args()

    df = generate_formatted_table(args.study_name, args.nt_file, args.r01_file, args.check, args.use_existing)
    generate_demographic_summary(args.study_name, df)
