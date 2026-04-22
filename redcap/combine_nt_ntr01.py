# -*- coding: utf-8 -*-
"""
Created on Fri May 24 20:32:15 2024

@author: grossens
"""

### Script to combine PreR01 data with NTR01 data 
    # accounts for variations in coding and adding additional fields for analysis
    
# import necessary modules
#import os
import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
from datetime import datetime
from pathlib import Path

# define user and current date
# use Path.home() so the script doesn't hard-code the Windows username
home = Path.home()
user = home.name
current_date = datetime.now().strftime("%Y_%m_%d")

## import spreadsheets from various data sources
# nt preR01 REDCap data
nt = pd.read_csv(home / 'Box' / 'Black_Lab' / 'projects' / 'TS' / 'NewTics' / 'Data' / 'Raw_Data' / 'REDCap backup' / 'NewTics_DATA_2024-09-05_1401.csv', low_memory=False)
# nt R01 REDCap data
r01 = pd.read_csv(home / 'Box' / 'Black_Lab' / 'projects' / 'TS' / 'New_Tics_R01' / 'Data' / 'analysis' / 'DS' / 'r01_combine' / 'NewTicsR01_DATA_2024-09-05_1400.csv', low_memory=False)
# weather data
weather = pd.read_csv(home / 'Box' / 'Black_Lab' / 'projects' / 'TS' / 'New_Tics_R01' / 'Data' / 'analysis' / 'DS' / 'r01_combine' / 'weather_result.csv')
# nt preR01 CBCL data
nt_cbcl = pd.read_csv(home / 'Box' / 'Black_Lab' / 'projects' / 'TS' / 'NewTics' / 'Data' / 'Raw_Data' / 'CBCL' / 'scores' / 'nt_cbcl_scores.csv')
nt_cbcl['redcap_event_name'] = nt_cbcl['redcap_event_name'].str.replace('12mo', '12_month_follow_up_arm_1')
nt_cbcl['redcap_event_name'] = nt_cbcl['redcap_event_name'].str.replace('screen', 'screening_visit_arm_1')
nt_cbcl['demo_study_id'] = nt_cbcl['demo_study_id'].str.upper()
# cpt data
cpt = pd.read_csv(home / 'Box' / 'Black_Lab' / 'projects' / 'TS' / 'New_Tics_R01' / 'Data' / 'analysis' / 'DS' / 'r01_combine' / 'cpt_result.csv')

## import file with information about how the NT and NTR01 REDCap fields overlap
REDCap_field_matching = pd.read_excel(home / 'Box' / 'Black_Lab' / 'projects' / 'TS' / 'New_Tics_R01' / 'Data' / 'analysis' / 'DS' / 'r01_combine' / 'matching_NT_and_NTR01_REDCap_fields.xlsx')

## make sure coding for the same data if coming from multiple possible forms (for example: srs* and srs_800*)
# maternal and familty history
mafh_cols_to_recode = ['mafh_threatened_abortion', 'mafh_twin_pregnancy', 'mafh_caf', 'mafh_smoke', 'mafh_alc', 'mafh_gestational_diabetes', \
                        'mafh_abn_vag_bleeding', 'mafh_hyperemesis', 'mafh_pre_eclampsia', 'mafh_maternal_malnutrition', 'mafh_iugr', \
                        'mafh_physical_abuse', 'mafh_preterm_labor', 'mafh_prem_or_prolon_rup', 'mafh_amnionitis', 'mafh_severe_pre_eclampsia', \
                        'mafh_breech_presentation', 'mafh_umb_cor_compress', 'mafh_fetal_distress', 'mafh_emerg_ces_section', 'mafh_traumatic_delivery', \
                        'mafh_birth_before_admiss', 'mafh_forceps_vacuum_used', 'mafh_meconium_aspiration', 'mafh_physical_injury', 'mafh_hypothermia', \
                        'mafh_hypoglycemia', 'mafh_neonatal_jaundice', 'mafh_respiratory_distress', 'mafh_pneumonia', 'mafh_intracranial_bleed', \
                        'mafh_necro_enterocolitis', 'mafh_ventilator', 'mafh_nicu_admission']
nt[mafh_cols_to_recode].replace({2:0}, inplace=True)
nt['mafh_if_yes_preterm'].replace([1, 2], [2, 0], inplace=True)
# KSADS
ksads_cols_to_recode = ['ksads_mdd_diag_curr_ep', 'ksads_mdd_prev_ep', 'ksads_dysth_cur_ep', 'ksads_dysth_prev_ep', 'ksads_ddnos_cur_ep', \
                        'ksads_ddnos_prev_ep', 'ksads_addm_cur_ep', 'ksads_addm_prev_ep', 'ksads_pd_cur_ep', 'ksads_pd_prev_ep', 'ksads_sad_cur_ep', \
                        'ksads_sad_prev_ep', 'ksads_adc_prev_ep', 'ksads_adc_cur_ep', 'ksads_sp_cur_ep', 'ksads_sp_prev_ep', 'ksads_socp_prev_ep', \
                        'ksads_socp_cur_ep', 'ksads_agor_prev_ep', 'ksads_agor_cur_ep', 'ksads_oad_prev_ep', 'ksads_oad_cur_ep', 'ksads_gad_cur_ep', \
                        'ksads_gad_prev_ep', 'ksads_ocd_cur_ep', 'ksads_ocd_prev_ep', 'ksads_ptsd_prev_ep', 'ksads_ptsd_cur_ep', 'ksads_asd_prev_ep', \
                        'ksads_asd_cur_ep', 'ksads_adam_cur_ep', 'ksads_adam_prev_ep', 'ksads_enur_cur_ep', 'ksads_enur_prev_ep', 'ksads_enco_cur_ep', \
                        'ksads_enco_prev_ep', 'ksads_an_cur_ep', 'ksads_an_prev_ep', 'ksads_bul_cur_ep', 'ksads_bul_prev_ep', 'ksads_add_prev_ep', \
                        'ksads_add_cur_ep', 'ksads_cd_cur_ep', 'ksads_cd_prev_ep', 'ksads_odd_cur_ep', 'ksads_odd_prev_ep', 'ksads_addc_cur_ep', \
                        'ksads_addc_prev_ep', 'ksads_admmc_cur_ep', 'ksads_admmc_prev_ep', 'ksads_ts_cur_ep', 'ksads_ts_prev_ep', 'ksads_cmvtd_cur_ep', \
                        'ksads_cmvtd_prev_ep', 'ksads_ttd_cur_ep', 'ksads_ttd_prev_ep', 'ksads_asperger_cur_ep', 'ksads_asperger_prev_ep', 'ksads_pddnos_prev_ep', \
                        'ksads_pddnos_cur_ep', 'ksads_aa_cur_ep', 'ksads_aa_prev_ep', 'ksads_ad_prev_ep', 'ksads_ad_cur_ep', 'ksads_sa_cur_ep', 'ksads_sa_prev_ep', \
                        'ksads_sd_prev_ep', 'ksads_sd_cur_ep', 'ksads_mr_prev_ep', 'ksads_opd_cur_ep', 'ksads_opd_prev_ep']
nt[ksads_cols_to_recode].replace([1, 2, 3, 5], [0, 1, 2, 3], inplace=True)
# DCI scores
dci_cols_to_recode = ['ts_dci_1', 'ts_dci_2', 'ts_dci_3', 'ts_dci_4', 'ts_dci_5', 'ts_dci_6', 'ts_dci_7', 'ts_dci_8', 'ts_dci_9', \
                      'ts_dci_10', 'ts_dci_11', 'ts_dci_12', 'ts_dci_13', 'ts_dci_14', 'ts_dci_15', 'ts_dci_16', 'ts_dci_17', 'ts_dci_18', \
                      'ts_dci_19', 'ts_dci_20', 'ts_dci_21', 'ts_dci_22', 'ts_dci_23', 'ts_dci_24', 'ts_dci_25', 'ts_dci_26', 'ts_dci_27']
nt[dci_cols_to_recode].replace({1:0}, inplace=True)
for x in range(1, 28):
    if x == 1:
        nt[f'ts_dci_{x}'].replace({2:15}, inplace=True)
    if x in (2, 3, 4):
        nt[f'ts_dci_{x}'].replace({2:5}, inplace=True)
    if x in (5, 8):
        nt[f'ts_dci_{x}'].replace({2:7}, inplace=True)
    if x in (9, 14, 15, 16, 19, 27):
        nt[f'ts_dci_{x}'].replace({2:1}, inplace=True)
    if x in (7, 13, 18, 22, 26):
        nt[f'ts_dci_{x}'].replace({2:4}, inplace=True)
    if x == 6:
        nt[f'ts_dci_{x}'].replace({2:12}, inplace=True)
# SRS scores
for sub in nt['demo_study_id']:
    if int(sub[2:]) > 800:
        for index in nt.loc[nt['demo_study_id'] == sub].index:
            for j in range(1, 66):
                nt.loc[index, f'srs_q{j}'] = nt.loc[index, f'srs_800_q{j}']
nt = nt.drop(columns=nt.filter(like='srs_800_q').columns)                
for x in range(1, 66):
    if x in (3, 7, 11, 12, 15, 17, 21, 22, 26, 32, 38, 40, 43, 45, 48, 52, 55):
        nt[f'srs_q{x}'].replace([1, 2, 3, 4], [3, 2, 1, 0], inplace=True)
    else:
        nt[f'srs_q{x}'].replace([1, 2, 3, 4], [0, 1, 2, 3], inplace=True)
srs_dic = {'srs_q1': 'mo', 'srs_q2': 'a', 'srs_q3': 'mo', 'srs_q4': 'ma', 'srs_q5': 'cog', 'srs_q6': 'mo', 'srs_q7': 'a', 'srs_q8': 'ma', 'srs_q9': 'mo', 'srs_q10': 'cog', 'srs_q11': 'mo', 'srs_q12': 'com',
          'srs_q13': 'com', 'srs_q14': 'ma', 'srs_q15': 'cog', 'srs_q16': 'com', 'srs_q17': 'cog', 'srs_q18': 'com', 'srs_q19': 'com', 'srs_q20': 'ma', 'srs_q21': 'com', 'srs_q22': 'com', 'srs_q23': 'mo', 'srs_q24': 'ma',
           'srs_q25': 'a', 'srs_q26': 'com', 'srs_q27': 'mo', 'srs_q28': 'ma', 'srs_q29': 'ma', 'srs_q30': 'cog', 'srs_q31': 'man', 'srs_q32': 'a', 'srs_q33': 'com', 'srs_q34': 'mo', 'srs_q35': 'com', 'srs_q36': 'com', 'srs_q37': 'com',
           'srs_q38': 'com', 'srs_q39': 'ma', 'srs_q40': 'cog', 'srs_q41': 'com', 'srs_q42': 'cog', 'srs_q43': 'mo', 'srs_q44': 'cog', 'srs_q45': 'a', 'srs_q46': 'com', 'srs_q47': 'com', 'srs_q48': 'cog', 'srs_q49': 'ma', 
           'srs_q50': 'ma', 'srs_q51': 'com', 'srs_q52': 'a', 'srs_q53': 'com', 'srs_q54': 'a', 'srs_q55': 'com', 'srs_q56': 'a', 'srs_q57': 'com', 'srs_q58': 'cog', 'srs_q59': 'cog', 'srs_q60': 'com', 'srs_q61': 'com', 'srs_q62': 'cog', 
           'srs_q63': 'ma', 'srs_q64': 'mo', 'srs_q65': 'mo'}

male_awareness_tdic = {0: 30, 1: 33, 2: 36, 3: 39, 4: 43, 5: 46, 6: 49, 7: 52, 8: 55, 9: 59, 10: 62, 11: 65, 12: 68, 13: 72, 14: 75, 15: 78, 16: 81, 17: 85, 18: 88}
female_awareness_tdic = {0: 30, 1: 34, 2: 38, 3: 41, 4: 45, 5: 48, 6: 52, 7: 55, 8: 59, 9: 63, 10: 66, 11: 70, 12: 73, 13: 77, 14: 80, 15: 84, 16: 88}
male_cognition_tdic = {0: 36, 1: 39, 2: 41, 3: 43, 4: 45, 5: 48, 6: 50, 7: 52, 8: 54, 9: 56, 10: 59, 11: 61, 12: 63, 13: 65, 14: 68, 15: 70, 16: 72, 17: 74, 18: 76, 19: 79, 20: 81, 21: 83, 22: 85, 23: 88}
female_cognition_tdic = {0: 38, 1: 40, 2: 42, 3: 45, 4: 47, 5: 50, 6: 52, 7: 55, 8: 57, 9: 60, 10: 62, 11: 64, 12: 67, 13: 69, 14: 72, 15: 74, 16: 77, 17: 79, 18: 81, 19: 84, 20: 86, 21: 89}
male_communication_tdic = {0: 36, 1: 37, 2: 38, 3: 39, 4: 41, 5: 42, 6: 43, 7: 45, 8: 46, 9: 47, 10: 48, 11: 50, 12: 51, 13: 52, 14: 53, 15: 55, 16: 56, 17: 57, 18: 58, 19: 60, 20: 61, 21: 62, 22: 64, 23: 65, 24: 66, 25: 67, 26: 69, 27: 70, 28: 71, 29: 72, 30: 74, 31: 75, 32: 76, 33: 77, 34: 79, 35: 80, 36: 81, 37: 83, 38: 84, 39: 85, 40: 86, 41: 88, 42: 89}
female_communication_tdic = {0: 37, 1: 38, 2: 40, 3: 41, 4: 43, 5: 44, 6: 46, 7: 47, 8: 49, 9: 50, 10: 52, 11: 53, 12: 55, 13: 56, 14: 58, 15: 59, 16: 60, 17: 62, 18: 63, 19: 65, 20: 66, 21: 68, 22: 69, 23: 71, 24: 72, 25: 74, 26: 75, 27: 77, 28: 78, 29: 80, 30: 81, 31: 83, 32: 84, 33: 85, 34: 87, 35: 88}
male_motivation_tdic = {0: 37, 1: 40, 2: 42, 3: 44, 4: 47, 5: 49, 6: 51, 7: 54, 8: 56, 9: 59, 10: 61, 11: 63, 12: 66, 13: 68, 14: 70, 15: 73, 16: 75, 17: 78, 18: 80, 19: 82, 20: 85, 21: 87}
female_motivation_tdic = {0: 38, 1: 40, 2: 43, 3: 45, 4: 48, 5: 50, 6: 53, 7: 55, 8: 58, 9: 60, 10: 62, 11: 65, 12: 67, 13: 70, 14: 72, 15: 75, 16: 77, 17: 80, 18: 82, 19: 84, 20: 87, 21: 89}
male_mannerisms_tdic = {0: 40, 1: 42, 2: 44, 3: 46, 4: 49, 5: 51, 6: 53, 7: 55, 8: 58, 9: 60, 10: 62, 11: 65, 12: 67, 13: 69, 14: 71, 15: 74, 16: 76, 17: 78, 18: 80, 19: 83, 20: 85, 21: 87}
female_mannerisms_tdic = {0: 41, 1: 44, 2: 46, 3: 49, 4: 52, 5: 55, 6: 58, 7: 61, 8: 64, 9: 67, 10: 70, 11: 73, 12: 76, 13: 79, 14: 82, 15: 85, 16: 88}

awareness = []
cognition = []
communication = []
motivation = []
mannerisms = []
for key, value in srs_dic.items():
    if value[0] == 'a':
        awareness.append(f'srs_{key[4:]}')
    elif value[0:3] == 'cog':
        cognition.append(f'srs_{key[4:]}')
    elif value[0:3] == 'com':
        communication.append(f'srs_{key[4:]}')
    elif value[0:2] == 'mo':
        motivation.append(f'srs_{key[4:]}')
    elif value[0:2] == 'ma':
        mannerisms.append(f'srs_{key[4:]}')
        
nt['srs_awareness'] = nt[awareness].sum(axis=1, min_count=1)
nt['srs_cognition'] = nt[cognition].sum(axis=1, min_count=1)
nt['srs_communication'] = nt[communication].sum(axis=1, min_count=1)
nt['srs_motivation'] = nt[motivation].sum(axis=1, min_count=1)
nt['srs_mannerisms'] = nt[mannerisms].sum(axis=1, min_count=1)

## then consilidate that data
for i in REDCap_field_matching[pd.isna(REDCap_field_matching['additional_NT_REDCap_field_1']) == False].index.values:
    for j in nt.index.values:
        if pd.isna(nt.loc[j, REDCap_field_matching.loc[i, 'NT_REDCap_fields']]) == True:
            if pd.isna(nt.loc[j, REDCap_field_matching.loc[i, 'additional_NT_REDCap_field_1']]) == False:
                nt.loc[j, REDCap_field_matching.loc[i, 'NT_REDCap_fields']] = nt.loc[j, REDCap_field_matching.loc[i, 'additional_NT_REDCap_field_1']]       
nt = nt.drop(columns = REDCap_field_matching[pd.isna(REDCap_field_matching['additional_NT_REDCap_field_1']) == False]['additional_NT_REDCap_field_1'])

## rename columns in NT data to match those in the NTR01 data
REDCap_field_matching_select = REDCap_field_matching[pd.isna(REDCap_field_matching['NTR01_REDCap_fields']) == False]
REDCap_field_matching_dict = dict(zip(REDCap_field_matching_select['NT_REDCap_fields'], REDCap_field_matching_select['NTR01_REDCap_fields']))   
nt = nt.rename(columns=REDCap_field_matching_dict)

# take care of any duplicate columns (while keeping the columns that have all of the data in them)
duplicate_columns = nt.columns[nt.columns.duplicated()].unique()
# Step 2: For each set of duplicate columns, retain the one with the most non-NaN values
for col in duplicate_columns:
    # Step 2a: Select columns with the duplicate name
    col_indices = nt.columns.get_loc(col)
    duplicate_df = nt.iloc[:, col_indices]
    
    # Step 2b: Count non-NaN values and find the column index with the maximum count
    non_nan_counts = duplicate_df.notna().sum()
    max_non_nan_col_idx = non_nan_counts.idxmax()
    
    # Step 2c: Drop other columns, keeping the one with the most non-NaN values
    cols_to_drop = [c for c in col_indices if c != max_non_nan_col_idx]
    nt.drop(nt.columns[cols_to_drop], axis=1, inplace=True)



# add coding for which database the data came from
nt.insert(loc=0, column='database', value=0)
r01.insert(loc=0, column='database', value=1)

### combine NT and NTR01 REDCap data
combined = pd.concat([nt, r01], axis=0).sort_values(by=['demo_study_id']).reset_index(drop=True)

# Drop any rows created for test/demo subjects whose study ID starts with 'test'
if 'demo_study_id' in combined.columns:
    test_mask = combined['demo_study_id'].astype(str).str.lower().str.startswith('test')
    if test_mask.any():
        combined = combined.loc[~test_mask].reset_index(drop=True)

# rename redcap event names to match
combined['redcap_event_name'] = combined['redcap_event_name'].str.replace("one_year_followup_arm_1", "12_month_follow_up_arm_1")
combined['redcap_event_name'] = combined['redcap_event_name'].str.replace("initial_screen_arm_1", "screening_visit_arm_1")

## add in weather data
combined = combined.merge(weather, on=('demo_study_id', 'redcap_event_name'), how='left')
weather_columns = ['weather_scores', 'weather_all', 'weather_block1', 'weather_block2', 'weather_block3', 'weather_block4', 'weather_block5', 'weather_block6', 'weather_block7', 'weather_block8', 'weather_block9', 'weather_block_data_complete']
for column in weather_columns:
    combined[column] = combined[f'{column}_y'].fillna(combined[f'{column}_x'])
    combined.drop([f'{column}_x', f'{column}_y'], axis=1, inplace=True)
    
## CBCL data
combined = combined.merge(nt_cbcl, on=('demo_study_id', 'redcap_event_name'), how='left')
nt_cbcl_post_merge = list(combined.columns)[-105:]
removelist = ['demo_study_id', 'redcap_event_name']
for entry in nt_cbcl_post_merge:
    if not entry.endswith("_y"):
        removelist.append(entry)
        
cbcl_columns = list(nt_cbcl.columns)
removed_cbcl = [c for c in cbcl_columns if c not in removelist]
for column in removed_cbcl:
    combined[column] = combined[f'{column}_y'].fillna(combined[f'{column}_x'])
    combined.drop([f'{column}_x', f'{column}_y'], axis=1, inplace=True)
    
ycbcl_default_unchecked = []
for x in range (1, 17):
    ycbcl_default_unchecked.append(f'ycbcl_clinical___{x}')
    ycbcl_default_unchecked.append(f'ycbcl_borderline___{x}')

for c in ycbcl_default_unchecked:
    combined[c] = np.where(~combined['ycbcl_t_attprob'].isnull(), combined[c], np.nan)
    
## CPT data
combined = combined.loc[:, ~combined.columns.str.startswith('cpt')]
combined = combined.merge(cpt, on=('demo_study_id', 'redcap_event_name'), how='left')

## SRS
combined['srs_awareness_t'] = combined.apply(lambda x: male_awareness_tdic[x['srs_awareness']] if not pd.isna(x['srs_awareness']) and (x['demo_sex'] == 1 and (x['srs_awareness'] >= 0 and x['srs_awareness'] < 19)) else x['srs_awareness'], axis=1)
combined['srs_awareness_t'] = combined.apply(lambda x: female_awareness_tdic[x['srs_awareness']] if not pd.isna(x['srs_awareness']) and (x['demo_sex'] == 0 and (x['srs_awareness'] >= 0 and x['srs_awareness'] < 17)) else x['srs_awareness_t'], axis=1)
combined['srs_awareness_t'] = combined.apply(lambda x: 90 if not pd.isna(x['srs_awareness']) and (x['srs_awareness'] >= 19 and x['demo_sex'] == 1) else x['srs_awareness_t'], axis=1)
combined['srs_awareness_t'] = combined.apply(lambda x: 90 if not pd.isna(x['srs_awareness']) and (x['srs_awareness'] >= 17 and x['demo_sex'] == 0) else x['srs_awareness_t'], axis=1)

combined['srs_cognition_t'] = combined.apply(lambda x: male_cognition_tdic[x['srs_cognition']] if not pd.isna(x['srs_cognition']) and (x['demo_sex'] == 1 and (x['srs_cognition'] >= 0 and x['srs_cognition'] < 24)) else x['srs_cognition'], axis=1)
combined['srs_cognition_t'] = combined.apply(lambda x: female_cognition_tdic[x['srs_cognition']] if not pd.isna(x['srs_cognition']) and (x['demo_sex'] == 0 and (x['srs_cognition'] >= 0 and x['srs_cognition'] < 22)) else x['srs_cognition_t'], axis=1)
combined['srs_cognition_t'] = combined.apply(lambda x: 90 if not pd.isna(x['srs_cognition']) and x['srs_cognition'] >= 24 else x['srs_cognition_t'], axis=1)
combined['srs_cognition_t'] = combined.apply(lambda x: 90 if not pd.isna(x['srs_cognition']) and (x['srs_cognition'] >= 22 and x['demo_sex'] == 0) else x['srs_cognition_t'], axis=1)

combined['srs_communication_t'] = combined.apply(lambda x: male_communication_tdic[x['srs_communication']] if not pd.isna(x['srs_communication']) and (x['demo_sex'] == 1 and (x['srs_communication'] >= 0 and x['srs_communication'] < 43)) else x['srs_communication'], axis=1)
combined['srs_communication_t'] = combined.apply(lambda x: female_communication_tdic[x['srs_communication']] if not pd.isna(x['srs_communication']) and (x['demo_sex'] == 0 and (x['srs_communication'] >= 0 and x['srs_communication'] < 36)) else x['srs_communication_t'], axis=1)
combined['srs_communication_t'] = combined.apply(lambda x: 90 if not pd.isna(x['srs_communication']) and x['srs_communication'] >= 43 else x['srs_communication_t'], axis=1)
combined['srs_communication_t'] = combined.apply(lambda x: 90 if not pd.isna(x['srs_communication']) and (x['srs_communication'] >= 36 and x['demo_sex'] == 0) else x['srs_communication_t'], axis=1)

combined['srs_motivation_t'] = combined.apply(lambda x: male_motivation_tdic[x['srs_motivation']] if not pd.isna(x['srs_motivation']) and (x['demo_sex'] == 1 and (x['srs_motivation'] >= 0 and x['srs_motivation'] < 22)) else x['srs_motivation'], axis=1)
combined['srs_motivation_t'] = combined.apply(lambda x: female_motivation_tdic[x['srs_motivation']] if not pd.isna(x['srs_motivation']) and (x['demo_sex'] == 0 and (x['srs_motivation'] >= 0 and x['srs_motivation'] < 22)) else x['srs_motivation_t'], axis=1)
combined['srs_motivation_t'] = combined.apply(lambda x: 90 if not pd.isna(x['srs_motivation']) and x['srs_motivation'] >= 22 else x['srs_motivation_t'], axis=1)
combined['srs_motivation_t'] = combined.apply(lambda x: 90 if not pd.isna(x['srs_motivation']) and (x['srs_motivation'] >= 22 and x['demo_sex'] == 0) else x['srs_motivation_t'], axis=1)

combined['srs_mannerisms_t'] = combined.apply(lambda x: male_mannerisms_tdic[x['srs_mannerisms']] if not pd.isna(x['srs_mannerisms']) and (x['demo_sex'] == 1 and (x['srs_mannerisms'] >= 0 and x['srs_mannerisms'] < 22)) else x['srs_mannerisms'], axis=1)
combined['srs_mannerisms_t'] = combined.apply(lambda x: female_mannerisms_tdic[x['srs_mannerisms']] if not pd.isna(x['srs_mannerisms']) and (x['demo_sex'] == 0 and (x['srs_mannerisms'] >= 0 and x['srs_mannerisms'] < 17)) else x['srs_mannerisms_t'], axis=1)
combined['srs_mannerisms_t'] = combined.apply(lambda x: 90 if not pd.isna(x['srs_mannerisms']) and x['srs_mannerisms'] >= 22 else x['srs_mannerisms_t'], axis=1)
combined['srs_mannerisms_t'] = combined.apply(lambda x: 90 if not pd.isna(x['srs_mannerisms']) and (x['srs_mannerisms'] >= 17 and x['demo_sex'] == 0) else x['srs_mannerisms_t'], axis=1)

combined['srs_total'] = combined['srs_awareness'] + combined['srs_cognition'] + combined['srs_communication'] + combined['srs_motivation'] + combined['srs_mannerisms']

male_total_tscore = {}
counter = 34
skips = [16, 39, 62, 87, 110]
for x in range (0, 117):
    male_total_tscore[x] = int(counter)
    if x not in skips: 
        counter += 0.5

female_total_tscore = {}
counter = 35
skips = [4, 15, 24, 33, 42, 53, 62, 71, 80, 91]
for x in range (0, 100):
    female_total_tscore[x] = int(counter)
    if x in skips:
        counter += 1
    else:
        counter += 0.5

combined['srs_total_t'] = combined.apply(lambda x: male_total_tscore[x['srs_total']] if (x['demo_sex'] == 1 and x['srs_total'] < 117 and not pd.isna(x['srs_total'])) else x['srs_total_t'], axis=1)
combined['srs_total_t'] = combined.apply(lambda x: female_total_tscore[x['srs_total']] if (x['demo_sex'] == 0 and x['srs_total'] < 100 and not pd.isna(x['srs_total'])) else x['srs_total_t'], axis=1)
combined['srs_total_t'] = combined.apply(lambda x: 90 if (x['demo_sex'] == 1 and x['srs_total'] >= 117 and not pd.isna(x['srs_total'])) else x['srs_total_t'], axis=1)
combined['srs_total_t'] = combined.apply(lambda x: 90 if (x['demo_sex'] == 0 and x['srs_total'] >= 100 and not pd.isna(x['srs_total'])) else x['srs_total_t'], axis=1)


## get group information
recruit_data = pd.read_csv(home / 'Box' / 'Black_Lab' / 'projects' / 'TS' / 'New_Tics_R01' / 'Data' / 'analysis' / 'DS' / 'r01_combine' / 'nih_recruitment.csv', \
                           encoding = 'latin1', sep='\t')
    
nt_group = recruit_data[['NewTicsID', 'TS_type', 'Group']].set_index('NewTicsID').T.to_dict('list')
def get_group(x):
    if x['demo_study_id'] in nt_group.keys():
        if nt_group[x['demo_study_id']][1] == 'TS':
            return nt_group[x['demo_study_id']][0]
        elif nt_group[x['demo_study_id']][1] == 'HC':
            return 'HC'
combined['nt_group'] = combined.apply(get_group, axis=1)
group = combined.pop('nt_group')
combined.insert(0,'nt_group', group)

# --- Determine current tic diagnosis fields per `reading_dx_from_redcap.md` rules ---
def determine_tic_dx_fields(row):
    # Map expert diagnosis codes to current diagnosis labels
    tourette = row.get('expert_diagnosis_tourette') if 'expert_diagnosis_tourette' in row.index else None
    chronic = row.get('expert_diagnosis_chronic_tics') if 'expert_diagnosis_chronic_tics' in row.index else None
    transient = row.get('expert_diagnosis_transient') if 'expert_diagnosis_transient' in row.index else None

    dx = np.nan
    # prefer 'present' (1) over 'subthreshold' (2)
    try:
        # first check for explicit present (1) in priority order
        if tourette == 1:
            dx = 'TS'
        elif chronic == 1:
            dx = 'CMTD'
        elif transient == 1:
            dx = 'PTD'
        else:
            # if none are '1', fall back to subthreshold (2) in same priority
            if tourette == 2:
                dx = 'TS-subthreshold'
            elif chronic == 2:
                dx = 'CMTD-subthreshold'
            elif transient == 2:
                dx = 'PTD-subthreshold'
    except Exception:
        dx = np.nan

    # Check control/inclusion flags if no diagnosis found
    if (pd.isna(dx) or dx is np.nan):
        # pre-R01 form name (if present) that indicated control enrollment
        if 'incl_excl_control' in row.index and row['incl_excl_control'] in (1, 2):
            dx = 'none'
        # R01 form name: incl_excl_grp where 3 == TFC (control)
        elif 'incl_excl_grp' in row.index and row['incl_excl_grp'] == 3:
            dx = 'none'

    # Determine criteria, source and date
    criteria = 'DSM-5' if not pd.isna(dx) else np.nan

    # prefer common candidate columns for clinician/source
    source_candidates = ['clinician', 'expert_diagnosis_clinician', 'tsp_rater', 'tsp_rater_name']
    source = np.nan
    for col in source_candidates:
        if col in row.index and pd.notna(row[col]):
            source = row[col]
            break

    # visit date from visit info
    visit_date = row.get('visit_date') if 'visit_date' in row.index else np.nan

    return pd.Series({
        'tic_diagnosis_current': dx,
        'tic_dx_current_criteria': criteria,
        'tic_dx_current_source': source,
        'tic_dx_current_date': visit_date
    })

# Apply the function and attach fields to `combined`
tic_dx_df = combined.apply(determine_tic_dx_fields, axis=1)
combined = pd.concat([combined, tic_dx_df], axis=1)

# Log rows that need human review (no dx and no clear control flag)
needs_review = combined[pd.isna(combined['tic_diagnosis_current']) & combined['demo_study_id'].notna()]
if len(needs_review) > 0:
    print(f"\n{len(needs_review)} visits need human review for tic diagnosis (no automatic mapping). Sample rows:")
    print(needs_review[['demo_study_id', 'redcap_event_name', 'incl_excl_grp', 'incl_excl_control', 'expert_diagnosis_tourette', 'expert_diagnosis_chronic_tics', 'expert_diagnosis_transient']].head(10))



# ### Choose "best" YGTSS score for each visit
# (post-TSP if available; otherwise pre-TSP)
combined['best_ygtss_impairment'] = combined.apply(lambda x: x['ygtss_past_week_expert_p_6'] if pd.isna(x['ygtss_post_drz_p_6']) else x['ygtss_post_drz_p_6'], axis=1)
combined['best_ygtss_tts'] = combined.apply(lambda x: x['ygtss_past_week_expert_total_tic_score'] if pd.isna(x['ygtss_post_drz_total_tic']) else x['ygtss_post_drz_total_tic'], axis=1)


# ### Edinburgh Handedness Inventory
def calculate_edinburgh(x):
    numerator = 0
    denominator = 0
    for q in edinburgh_questions:
        if not pd.isna(x[q]):
            if x[q] == -2:
                numerator += -2
                denominator += 2
            elif x[q] == -1:
                numerator += -1
                denominator += 1
            elif x[q] == 4:
                denominator += 2
            elif x[q] == 1:
                numerator += 1
                denominator += 1
            elif x[q] == 2:
                numerator += 2
                denominator += 2
    return (np.float64(numerator)/denominator)*100

edinburgh_questions = []
for x in range (1, 11):
    edinburgh_questions.append(f'edinburgh_handedness_{x}')
combined['ehi_laterality_quotient'] = combined.apply(calculate_edinburgh, axis =1)
combined['ehi_laterality_quotient'] = combined['ehi_laterality_quotient'].apply(lambda x: round(x, 2) if not pd.isna(x) else x)

def calculate_ehi_handedness(x):
    if not pd.isna(x['ehi_laterality_quotient']):
        if x['ehi_laterality_quotient'] >= 60:
            return 'R' 
        elif x['ehi_laterality_quotient'] <= -60:
            return 'L'
        else:
            return 'A'

combined['ehi_handedness'] = combined.apply(calculate_ehi_handedness, axis=1)

def calculate_peg_handedness(x):
    if not pd.isna(x['ehi_laterality_quotient']):
        if x['ehi_laterality_quotient'] >= 0:
            return 'R'
        else:
            return 'L'

combined['peg_handedness'] = combined.apply(calculate_peg_handedness, axis=1)
combined['peg_score'] = combined.apply(lambda x: x['peg_right_30s'] if x['peg_handedness'] == 'R' else x['peg_left_30s'], axis=1)

# --- Compute `race` column from demo_race___1..7 (based on screening visit) ---
race_cols = [f'demo_race___{i}' for i in range(1, 8)]
# helper to determine race from a single row (screening)
def _race_from_row(row):
    vals = []
    for c in race_cols:
        if c in row.index:
            try:
                v = row[c]
            except Exception:
                v = None
        else:
            v = None
        # treat NaN as 0 / missing
        if pd.isna(v):
            vals.append(0)
        else:
            try:
                vals.append(int(v))
            except Exception:
                # non-integer values -> treat as 0
                vals.append(0)

    # count positive selections
    positives = sum(1 for v in vals if v > 0)
    if positives > 1:
        return 'More than one race'
    # map single selection
    mapping = {
        1: 'American Indian/Alaska Native',
        2: 'Asian',
        3: 'Hawaiian or Pacific Islander',
        4: 'Black',
        5: 'White',
        6: 'More than one race',
        7: 'Unknown'
    }
    for idx, v in enumerate(vals, start=1):
        if v > 0:
            return mapping.get(idx, 'Unknown')
    # no selection
    return 'Unknown or not reported'

# build mapping of demo_study_id -> race using screening visit
race_map = {}
if 'demo_study_id' in combined.columns:
    screening = combined.loc[combined['redcap_event_name'] == 'screening_visit_arm_1'] if 'redcap_event_name' in combined.columns else combined.iloc[0:0]
    for sid, group in screening.groupby('demo_study_id'):
        # try to find first non-empty race from screening rows for subject
        assigned = None
        for _, r in group.iterrows():
            assigned = _race_from_row(r)
            if assigned is not None:
                break
        race_map[sid] = assigned if (assigned is not None) else 'Unknown or not reported'

# default for subjects without screening entry
default_race = 'Unknown or not reported'

# map to combined; ensure all rows get a race value copied from screening
combined['race'] = combined['demo_study_id'].map(race_map).fillna(default_race)

# Insert `race` column immediately after `ehi_handedness` (or after `peg_handedness` if not present)
cols = list(combined.columns)
if 'race' in cols:
    cols.remove('race')
insert_pos = None
if 'ehi_handedness' in cols:
    insert_pos = cols.index('ehi_handedness') + 1
elif 'handedness' in cols:
    insert_pos = cols.index('handedness') + 1
elif 'peg_handedness' in cols:
    insert_pos = cols.index('peg_handedness') + 1
else:
    insert_pos = len(cols)
# reorder columns to place race at desired position
cols.insert(insert_pos, 'race')
combined = combined.loc[:, cols]

# --- Compute `ethnicity` column from demo_ethnicity___1..3 (based on screening visit) ---
eth_cols = [f'demo_ethnicity___{i}' for i in range(1, 4)]
def _eth_from_row(row):
    vals = []
    for c in eth_cols:
        if c in row.index:
            try:
                v = row[c]
            except Exception:
                v = None
        else:
            v = None
        if pd.isna(v):
            vals.append(0)
        else:
            try:
                vals.append(int(v))
            except Exception:
                vals.append(0)

    positives = sum(1 for v in vals if v > 0)
    # if multiple selections, treat as Unknown
    if positives > 1:
        return 'Unknown'
    mapping = {
        1: 'Not Hispanic or Latino',
        2: 'Hispanic or Latino',
        3: 'Unknown'
    }
    for idx, v in enumerate(vals, start=1):
        if v > 0:
            return mapping.get(idx, 'Unknown')
    return 'Unknown'

# build mapping of demo_study_id -> ethnicity using screening visit
eth_map = {}
if 'demo_study_id' in combined.columns:
    screening = combined.loc[combined['redcap_event_name'] == 'screening_visit_arm_1'] if 'redcap_event_name' in combined.columns else combined.iloc[0:0]
    for sid, group in screening.groupby('demo_study_id'):
        assigned = None
        for _, r in group.iterrows():
            assigned = _eth_from_row(r)
            if assigned is not None:
                break
        eth_map[sid] = assigned if (assigned is not None) else 'Unknown'

default_eth = 'Unknown'
combined['ethnicity'] = combined['demo_study_id'].map(eth_map).fillna(default_eth)

# insert `ethnicity` immediately after `race` if possible
cols = list(combined.columns)
if 'ethnicity' in cols:
    cols.remove('ethnicity')
if 'race' in cols:
    pos = cols.index('race') + 1
else:
    pos = len(cols)
cols.insert(pos, 'ethnicity')
combined = combined.loc[:, cols]


# ### Days since onset
def calculate_days_onset(x):
    if x['demo_study_id'] in nt_group.keys():
        if x['redcap_event_name'] == 'screening_visit_arm_1':
            return recruit_data.loc[(recruit_data['NewTicsID'] == x['demo_study_id']), 'days_since_onset_screen'].values[0].astype('timedelta64[D]').astype(int)
        elif x['redcap_event_name'] == '12_month_follow_up_arm_1':
            return recruit_data.loc[(recruit_data['NewTicsID'] == x['demo_study_id']), 'days_since_onset_12mo'].values[0].astype('timedelta64[D]').astype(int)

recruit_data["days_since_onset_screen"] = recruit_data.apply(lambda x: datetime.strptime(x["Text11"], "%m/%d/%Y") - datetime.strptime(x["TicOnset"], "%m/%d/%Y") if not pd.isna(x['Text11']) and not pd.isna(x['TicOnset']) else np.nan, axis=1)
recruit_data["days_since_onset_12mo"] = recruit_data.apply(lambda x: datetime.strptime(x["Text30"], "%m/%d/%Y") - datetime.strptime(x["TicOnset"], "%m/%d/%Y") if not pd.isna(x['Text30']) and not pd.isna(x['TicOnset']) else np.nan, axis=1)

combined['days_since_onset'] = combined.apply(calculate_days_onset, axis=1)


# ### Anxiety disorders
anxiety_disorders = ['panic', 'agoraphobia', 'sepanxiety', 'socanxiety', 'mutism', 'gad', 'anxdo_uns', 'adjdo_anx']
anxiety_columns = []
for disorder in anxiety_disorders:
    anxiety_columns.append(f'ksads5_{disorder}_ce')
    anxiety_columns.append(f'ksads5_{disorder}_msp')

def calculate_anxiety(x):
    anxiety_present = False
    value_present = False
    for c in anxiety_columns:
        if not pd.isna(x[c]):
            value_present = True
        if x[c] == 2 or x[c] == 3 or x[c] == 4:
            anxiety_present = True
            break
    if value_present:
        return anxiety_present
    elif not pd.isna(x['ksads5_anxiety_yn']):
        if x['ksads5_anxiety_yn'] == 1:
            return True
        elif x['ksads5_anxiety_yn'] == 0:
            return False
combined['anxiety_disorder_ever'] = combined.apply(calculate_anxiety, axis=1)


# ### OCD
ocd_columns = ['ksads5_ocd_ce', 'ksads5_ocd_msp']
def calculate_ocd(x):
    ocd_present = False
    for c in ocd_columns:
        if x[c] == 2 or x[c] == 3 or x[c] == 4:
            ocd_present = True
            break
    if not pd.isna(x['ksads5_ocd_ce']) or not pd.isna(x['ksads5_ocd_msp']):
        return ocd_present
combined['ocd_ever'] = combined.apply(calculate_ocd, axis=1)


# ### ADHD
adhd_columns = ['ksads_add_prev_ep', 'ksads_add_cur_ep']
def calculate_adhd(x):
    if not pd.isna(x['ksads5_adhd_yn']):
        if x['ksads5_adhd_yn'] == 1:
            return True
        elif x['ksads5_adhd_yn'] == 0:
            return False
    elif not pd.isna(x['ksads_add_prev_ep']) or not pd.isna(x['ksads_add_cur_ep']):
        adhd_present = False
        for c in adhd_columns:
            if x[c] == 3 or x[c] == 4 or x[c] == 5:
                adhd_present = True
                break
        return adhd_present
combined['adhd_ever'] = combined.apply(calculate_adhd, axis=1)


# ### PedsQL
pedsql_phys = []
for x in range(1, 9):
    pedsql_phys.append(f'pedsql_phys_{x}')
pedsql_emo = []
for x in range(1, 6):
    pedsql_emo.append(f'pedsql_emo_{x}')
pedsql_social = []
for x in range(1, 6):
    pedsql_social.append(f'pedsql_social_{x}')
pedsql_school = []
for x in range(1, 6):
    pedsql_school.append(f'pedsql_school_{x}')
pedsql_columns = pedsql_phys + pedsql_emo + pedsql_social + pedsql_school
for pedsql in pedsql_columns:
    combined[pedsql] = combined[pedsql].apply(lambda x: x-1 if not pd.isna(x) else x)
pedsql_psychosocial = pedsql_emo + pedsql_social + pedsql_school

def pedsql_phys_scaled(x):
    scale = np.nan
    counter = 0
    ped_sum = 0
    for phys in pedsql_phys:
        if not pd.isna(x[phys]):
            counter += 1
            ped_sum += x[phys]
    if counter >= 4:
        scale = 100 - 25 * (ped_sum / counter)
    return scale

combined['pedql_phys_scaled'] = combined.apply(pedsql_phys_scaled, axis=1)

def pedsql_emo_scaled(x):
    scale = np.nan
    counter = 0
    ped_sum = 0
    for phys in pedsql_emo:
        if not pd.isna(x[phys]):
            counter += 1
            ped_sum += x[phys]
    if counter >= 3:
        scale = 100 - 25 * (ped_sum / counter)
    return scale
combined['pedql_emo_scaled'] = combined.apply(pedsql_emo_scaled, axis=1)

def pedsql_social_scaled(x):
    scale = np.nan
    counter = 0
    ped_sum = 0
    for phys in pedsql_social:
        if not pd.isna(x[phys]):
            counter += 1
            ped_sum += x[phys]
    if counter >= 3:
        scale = 100 - 25 * (ped_sum / counter)
    return scale
combined['pedql_social_scaled'] = combined.apply(pedsql_social_scaled, axis=1)

def pedsql_school_scaled(x):
    scale = np.nan
    counter = 0
    ped_sum = 0
    for phys in pedsql_school:
        if not pd.isna(x[phys]):
            counter += 1
            ped_sum += x[phys]
    if counter >= 3:
        scale = 100 - 25 * (ped_sum / counter)
    return scale
combined['pedql_school_scaled'] = combined.apply(pedsql_school_scaled, axis=1)

def pedsql_psychosocial_scaled(x):
    scale = np.nan
    counter = 0
    ped_sum = 0
    for phys in pedsql_psychosocial:
        if not pd.isna(x[phys]):
            counter += 1
            ped_sum += x[phys]
    if counter >= 8:
         scale = 100 - 25 * (ped_sum / counter)
    return scale
combined['pedsql_psychosocial_scaled'] = combined.apply(pedsql_psychosocial_scaled, axis=1)

def pedsql_total_scaled(x):
    scale = np.nan
    counter = 0
    ped_sum = 0
    for phys in pedsql_columns:
        if not pd.isna(x[phys]):
            counter += 1
            ped_sum += x[phys]
    if counter >= 12:
         scale = 100 - 25 * (ped_sum / counter)
    return scale
combined['pedsql_total_scaled'] = combined.apply(pedsql_total_scaled, axis=1)


# ### Tics in neck?
tic_neck = combined[(combined['ts_dci_score_9'] == 1) & (combined['ts_dci_score_12'] == 0)]
t_neck = tic_neck[['demo_study_id', 'redcap_event_name', 'drz_tics_bf_bselne', 'drz_tics_1st_bselne', 'drz_tics_aftr_1st_bselne']]
manual_rate_neck_tic = pd.read_excel(home / 'Box' / 'Black_Lab' / 'projects' / 'TS' / 'New_Tics_R01' / 'Data' / 'analysis' / 'DS' / 'r01_combine' / 'tics_below_neck.xlsx')
combined = combined.merge(manual_rate_neck_tic, on=('demo_study_id', 'redcap_event_name', 'drz_tics_bf_bselne', 'drz_tics_1st_bselne', 'drz_tics_aftr_1st_bselne' ), how='left')
combined.drop(columns=['Unnamed: 0'], inplace=True)

exp_tic_list_check = ['exp_shoulder_jerks___1', 'exp_shoulder_jerks_2___1', 'exp_arm_movements___1', 'exp_comb_hair___1', 'exp_leg_mvmts___1', 'exp_leg_mvmts2_b66___1', 'exp_ab_tensing___1', 'exp_touching___1', 'exp_touching2_157___1', 'exp_touching2_1572_d79___1', 'exp_touching2_1572_d792_311___1',
                     'exp_touching2_1572_d792_32_51f___1', 'exp_touching2_1572_d792_32_e3f___1', 'exp_touching2_1572_d792_32_bf0___1', 'exp_touching2_1572_d792_32_71d___1']

def check_exp_tic_list(x):
    for column in exp_tic_list_check:
        if x[column] == 1:
            return True

combined['exp_tic_below_neck'] = combined.apply(check_exp_tic_list, axis=1)

def tic_below_neck(x):
    if x['ts_dci_score_9'] == 0:
        return True
    if x['ts_dci_score_12'] == 1:
        return True
    if x['exp_tic_below_neck'] == 1:
        return True
    if x['below_neck_from_expert_tic_list'] == 1:
        return True
    if x['below_neck_from_old_TSP_notes'] == 1:
        return True
    if x['below_neck_from_TSP_form'] == 1:
        return True

# **Only time `tic_below_neck` should be blank is if every one of those forms is blank.**
combined['tic_below_neck'] = combined.apply(tic_below_neck, axis=1)


# ### TSP
tic_timer = pd.read_csv(home / 'Box' / 'Black_Lab' / 'projects' / 'TS' / 'New_Tics_R01' / 'Data' / 'TSP' / 'tic_timer_data.csv')
data = {'demo_study_id': [], 'redcap_event_name':[], 'baseline_tic_freq': [], 'verbal_tic_freq': [], 'drz_tic_freq': [], 'ncr_tic_freq' :[],
        'baseline_reward_freq': [], 'verbal_reward_freq': [], 'drz_reward_freq': [], 'ncr_reward_freq': []}

def calculate_freq(n, d):
    return n / d if d else np.nan

current_subj = 'NT701'
current_event = '12_month_follow_up_arm_1'
baseline_tic = 0
baseline_reward = 0
baseline_duration = 0
verbal_tic = 0
verbal_reward = 0
verbal_duration = 0
drz_tic = 0
drz_reward = 0
drz_duration = 0
ncr_tic = 0
ncr_reward = 0
ncr_duration = 0
ncr_check = False
tic_timer_calc = pd.DataFrame()
tic_timer_calc['demo_study_id'] = []
tic_timer_calc['redcap_event_name'] = []
tic_timer_calc['baseline_tic_freq'] = []
tic_timer_calc['verbal_tic_freq'] = []
tic_timer_calc['drz_tic_freq'] = []
tic_timer_calc['ncr_tic_freq'] = []
tic_timer_calc['baseline_10s_freq'] = []
tic_timer_calc['verbal_10s_freq'] = []
tic_timer_calc['drz_10s_freq'] = []
tic_timer_calc['ncr_10s_freq'] = []
last_index = tic_timer.last_valid_index()
for index, row in tic_timer.iterrows():
    if current_subj == row['demo_study_id'] and current_event == row['redcap_event_name']:
        date = row['date']
        if row['condition'] == 'baseline':
            baseline_tic += row['tics']
            baseline_reward += row['rewards']
            baseline_duration += row['duration']
        elif row['condition'] == 'verbal':
            verbal_tic += row['tics']
            verbal_reward += row['rewards']
            verbal_duration += row['duration']
        elif row['condition'] == 'DRZ':
            drz_tic += row['tics']
            drz_reward += row['rewards']
            drz_duration += row['duration']
        elif row['condition'] == 'NCR':
            ncr_check = True
            ncr_tic += row['tics']
            ncr_reward += row['rewards']
            ncr_duration += row['duration']
        if index == last_index:
            if not ncr_check:
                ncr_tic = np.nan
                ncr_reward = np.nan
                ncr_duration = 1
            subj = {'demo_study_id': current_subj, 'redcap_event_name': current_event, 'baseline_tic_freq': calculate_freq(baseline_tic, baseline_duration), 'verbal_tic_freq': calculate_freq(verbal_tic, verbal_duration),
               'drz_tic_freq': calculate_freq(drz_tic, drz_duration), 'ncr_tic_freq': calculate_freq(ncr_tic, ncr_duration), 'baseline_10s_freq': calculate_freq(baseline_reward, baseline_duration), 'verbal_10s_freq': calculate_freq(verbal_reward,verbal_duration), 'drz_10s_freq': calculate_freq(drz_reward, drz_duration), 'ncr_10s_freq': calculate_freq(ncr_reward, ncr_duration), 'tsp_date': date, 'tsp_rater': 'live'}
            tic_timer_calc = pd.concat([tic_timer_calc, pd.DataFrame([subj])], ignore_index=True)
    else:
        subj = {'demo_study_id': current_subj, 'redcap_event_name': current_event, 'baseline_tic_freq': calculate_freq(baseline_tic, baseline_duration), 'verbal_tic_freq': calculate_freq(verbal_tic, verbal_duration), 'drz_tic_freq': calculate_freq(drz_tic, drz_duration), 'ncr_tic_freq': calculate_freq(ncr_tic, ncr_duration), 'baseline_10s_freq': calculate_freq(baseline_reward, baseline_duration), 'verbal_10s_freq': calculate_freq(verbal_reward,verbal_duration), 'drz_10s_freq': calculate_freq(drz_reward, drz_duration), 'ncr_10s_freq': calculate_freq(ncr_reward, ncr_duration), 'tsp_date': date, 'tsp_rater': 'live'}
        tic_timer_calc = pd.concat([tic_timer_calc, pd.DataFrame([subj])], ignore_index=True)
        current_subj = row['demo_study_id']
        current_event = row['redcap_event_name']
        baseline_tic = 0
        baseline_reward = 0
        baseline_duration = 0
        verbal_tic = 0
        verbal_reward = 0
        verbal_duration = 0
        drz_tic = 0
        drz_reward = 0
        drz_duration = 0
        ncr_tic = 0
        ncr_reward = 0
        ncr_duration = 0
        ncr_check = False
        if row['condition'] == 'baseline':
            baseline_tic += row['tics']
            baseline_reward += row['rewards']
            baseline_duration += row['duration']
        elif row['condition'] == 'verbal':
            verbal_tic += row['tics']
            verbal_reward += row['rewards']
            verbal_duration += row['duration']
        elif row['condition'] == 'DRZ':
            drz_tic += row['tics']
            drz_reward += row['rewards']
            drz_duration += row['duration']
        elif row['condition'] == 'NCR':
            ncr_tic += row['tics']
            ncr_reward += row['rewards']
            ncr_duration += row['duration']

combined = combined.merge(tic_timer_calc, on=('demo_study_id', 'redcap_event_name'), how='left')


# ### Barratt's SES score
combined['barratt_education_score'] = combined[['ses_edu_level_mother', 'ses_edu_level_father', 'ses_edu_level_mpartner', 'ses_edu_level_fpartner']].mean(axis=1)
combined['barratt_occupation_score'] = combined[['ses_occ_mother', 'ses_occ_father', 'ses_occ_mpartner', 'ses_occ_fpartner']].mean(axis=1)
combined['barratt_ses_score'] = combined['barratt_education_score'] + combined['barratt_occupation_score']



# ### Get rid of extra (arbitrary) rows - filled with zeros and no visit date
subjects = combined['demo_study_id'].drop_duplicates()
all_duplicated = []
all_duplicated_no_date = []
for subject in subjects:
    duplicated_by_subject = combined.groupby(combined['demo_study_id']).get_group(subject)[((combined.groupby(combined['demo_study_id']).get_group(subject).duplicated(subset = ['redcap_event_name'], keep=False)) == True)]
    all_duplicated.append(duplicated_by_subject)
all_duplicated_better = pd.concat(all_duplicated)
duplicated_no_date = all_duplicated_better[(all_duplicated_better['visit_date'].str.contains('0', na = False) == False)]
combined = combined.drop(labels=duplicated_no_date.index, axis = 0)


# ### Add clinically_meaningful column
def clinically_meaningful(x):
    if x['best_ygtss_impairment'] > 13:
        return True
    if x['best_ygtss_tts'] >= 20:
        return True
    if x['outcome_data_4'] == 2: # this is the coding straight from REDCap (no is 1, yes is 2)
        return True
    if x['outcome_data_7'] == 2:
        return True
    return False
combined['clinically_meaningful'] = combined.apply(clinically_meaningful, axis=1)


# ### Add columns for comorbid psychiatric diagnoses
combined['comorbid_psychiatric_diag'] = np.full(len(combined), 0)

for x in combined.index.values:
    if combined.loc[x, 'ksads5_asd_ce'] == 3:
        combined.loc[x, 'comorbid_psychiatric_diag'] = combined.loc[x, 'comorbid_psychiatric_diag'] + 1
    if (combined.loc[x, 'expert_diagnosis_adhd'] == 1) | (combined.loc[x, 'adhd_ever'] == True):
        combined.loc[x, 'comorbid_psychiatric_diag'] = combined.loc[x, 'comorbid_psychiatric_diag'] + 1
    if (combined.loc[x, 'ksads5_cd_ce'] == 3) | (combined.loc[x, 'ksads5_cd_msp'] == 3):
        combined.loc[x, 'comorbid_psychiatric_diag'] = combined.loc[x, 'comorbid_psychiatric_diag'] + 1
    if (combined.loc[x, 'anxiety_disorder_ever'] == True) | (combined.loc[x, 'ksads5_anxiety_yn'] == 1) |        (combined.loc[x, 'ksads5_specphobia_ce'] == 3) | (combined.loc[x,  'ksads5_specphobia_msp'] == 3) |        (combined.loc[x, 'ksads5_agoraphobia_ce'] == 3) | (combined.loc[x, 'ksads5_agoraphobia_msp'] == 3) |         (combined.loc[x, 'ksads5_agoraphobia_ce'] == 3) | (combined.loc[x, 'ksads5_agoraphobia_msp'] == 3) |         (combined.loc[x, 'ksads5_mddaddm_ce'] == 3) | (combined.loc[x, 'ksads5_mddaddm_msp'] == 3) |         (combined.loc[x, 'ksads5_adjdo_anx_ce'] == 3) | (combined.loc[x, 'ksads5_adjdo_anx_msp'] == 3) |         (combined.loc[x, 'ksads5_adjdo_doc_ce'] == 3) | (combined.loc[x, 'ksads5_adjdo_doc_msp'] == 3) |         (combined.loc[x, 'ksads5_adjdo_mix_ce'] == 3) | (combined.loc[x, 'ksads5_adjdo_mix_msp'] == 3):
        combined.loc[x, 'comorbid_psychiatric_diag'] = combined.loc[x, 'comorbid_psychiatric_diag'] + 1
    if (combined.loc[x, 'expert_diagnosis_ocd'] == 1) | (combined.loc[x, 'ocd_ever'] == True):
        combined.loc[x, 'comorbid_psychiatric_diag'] = combined.loc[x, 'comorbid_psychiatric_diag'] + 1
    if (combined.loc[x, 'ksads5_mdd_ce'] == 3) | (combined.loc[x, 'ksads5_mdd_msp'] == 3) |         (combined.loc[x, 'ksads5_dysth_ce'] == 3) | (combined.loc[x, 'ksads5_dysth_msp'] == 3) |         (combined.loc[x, 'ksads5_unspecmood_ce'] == 3) | (combined.loc[x, 'ksads5_unspecmood_msp'] == 3):
        combined.loc[x, 'comorbid_psychiatric_diag'] = combined.loc[x, 'comorbid_psychiatric_diag'] + 1
    if (combined.loc[x, 'ksads5_unspecbipolar_ce'] == 3) | (combined.loc[x, 'ksads5_unspecbipolar_msp'] == 3) |         (combined.loc[x, 'ksads5_schizophrenia_ce'] == 3) | (combined.loc[x, 'ksads5_schizophrenia_msp'] == 3):
        combined.loc[x, 'comorbid_psychiatric_diag'] = combined.loc[x, 'comorbid_psychiatric_diag'] + 1
  

# ### Add column for total lifetime tics
for i in combined.index.values:
    if pd.isna(combined.loc[i, 'drz_tics_bf_bselne']) == False:
        tics_before_baseline = (str(combined.loc[i, 'drz_tics_bf_bselne']).count('  ')+1)
        # print('\n tics_before_baseline=',combined.loc[i, 'demo_study_id'], tics_before_baseline)
    else:
        tics_before_baseline = 0
        # print('\n tics_before_baseline=',combined.loc[i, 'demo_study_id'], tics_before_baseline)
    if pd.isna(combined.loc[i, 'drz_tics_1st_bselne']) == False:
        tics_at_baseline = (str(combined.loc[i, 'drz_tics_1st_bselne']).count('  ')+1)
        # print('drz_tics_1st_bselne=',combined.loc[i, 'demo_study_id'], tics_at_baseline)
    else:
        tics_at_baseline = 0
        # print('drz_tics_1st_bselne=',combined.loc[i, 'demo_study_id'], tics_at_baseline)
    if pd.isna(combined.loc[i, 'drz_tics_aftr_1st_bselne']) == False:
        tics_after_baseline = (str(combined.loc[i, 'drz_tics_aftr_1st_bselne']).count('  ')+1)
        # print('drz_tics_aftr_1st_bselne=',combined.loc[i, 'demo_study_id'], tics_after_baseline)
        # print('TOTAL=',combined.loc[i, 'demo_study_id'], tics_after_baseline+tics_at_baseline+tics_before_baseline)
    else:
        tics_after_baseline = 0
        # print('drz_tics_aftr_1st_bselne=',combined.loc[i, 'demo_study_id'], tics_after_baseline)
        # print('TOTAL=',combined.loc[i, 'demo_study_id'], tics_after_baseline+tics_at_baseline+tics_before_baseline)
        # before bsline =1 
    combined.loc[i, 'before_baseline_tics'] = tics_before_baseline+tics_at_baseline+tics_after_baseline
    combined.loc[i, 'after_baseline_tics'] = tics_before_baseline+tics_at_baseline+tics_after_baseline
    combined.loc[i, 'total_lifetime_tics'] = tics_before_baseline+tics_at_baseline+tics_after_baseline

for i in combined.index.values:
    if pd.isna(combined.loc[i, 'drz_tics_bf_bselne']) == False:
        tics_before_baseline = (str(combined.loc[i, 'drz_tics_bf_bselne']).count('  ')+1)
    else:
        tics_before_baseline = 0
    if pd.isna(combined.loc[i, 'drz_tics_1st_bselne']) == False:
        tics_at_baseline = (str(combined.loc[i, 'drz_tics_1st_bselne']).count('  ')+1)
    else:
        tics_at_baseline = 0
    if pd.isna(combined.loc[i, 'drz_tics_aftr_1st_bselne']) == False:
        tics_after_baseline = (str(combined.loc[i, 'drz_tics_aftr_1st_bselne']).count('  ')+1)
    else:
        tics_after_baseline = 0
    combined.loc[i, 'total_lifetime_tics'] = tics_before_baseline+tics_at_baseline+tics_after_baseline


# ### Add CBCL DECR Column
combined['CBCL_DECR'] = combined['cbcl_t_anxdepr'] + combined['cbcl_t_attenprob'] + combined['cbcl_t_aggbeh']

### Add column to specify what visit the data should be analyzed with (in the case of instances where data is collected 
# multiple years post tic onset, but recorded at the 12mo visit to ensure all of the things were collected)
for i in combined.index.values:
    if (combined.loc[i, 'demo_study_id'] == 'NT870') & (combined.loc[i, 'redcap_event_name'] == '12_month_follow_up_arm_1'):
        combined.loc[i, 'event_name_for_analysis'] = 'clinical_follow_up_arm_1c'
    if (combined.loc[i, 'demo_study_id'] == 'NT933') & (combined.loc[i, 'redcap_event_name'] == '12_month_follow_up_arm_1'):
        combined.loc[i, 'event_name_for_analysis'] = 'clinical_follow_up_arm_1'
    else:
        combined.loc[i, 'event_name_for_analysis'] = combined.loc[i, 'redcap_event_name']
        
# ### Add column for "Perfect_SATS"
for i in combined.index.values:
    # if a participant has information for both outcome data #5 and #9
    if (pd.isna(combined.loc[i, 'outcome_data_5']) == False) & (pd.isna(combined.loc[i, 'outcome_data_9']) == False):
        if (combined.loc[i, 'outcome_data_5'] == 1) & (combined.loc[i, 'outcome_data_9'] == 2):
            combined.loc[i, 'Perfect_SATS'] = True
        else:
            combined.loc[i, 'Perfect_SATS'] = False
                        
# ### Add columns for DSM-5 onset dates (which are only different for about three people - NT714, NT734, and NT806)
participants_with_complicated_onset_dates = ['NT714', 'NT734', 'NT806']
combined['dsm-5_expert_diagnosis_onset_date'] = combined['expert_diagnosis_onset'].copy()
combined['dsm-5_expert_diagnosis_onset_confid'] = combined['expert_diagnosis_onset_confid'].copy()
for i in combined.index.values:
    if combined.loc[i, 'demo_study_id'] == 'NT714':
        combined.loc[i, 'dsm-5_expert_diagnosis_onset_date'] = pd.to_datetime('6/7/2009')
        combined.loc[i, 'dsm-5_expert_diagnosis_onset_confid'] = "06/07/2008 - 06/07/2010"
    elif combined.loc[i, 'demo_study_id'] == 'NT734':
        combined.loc[i, 'dsm-5_expert_diagnosis_onset_date'] = pd.to_datetime('4/15/2011')
        combined.loc[i, 'dsm-5_expert_diagnosis_onset_confid'] = "04/01/2011 - 02/28/2012"
    elif combined.loc[i, 'demo_study_id'] == 'NT806':
        combined.loc[i, 'dsm-5_expert_diagnosis_onset_date'] = pd.to_datetime('1/15/2014')
        combined.loc[i, 'dsm-5_expert_diagnosis_onset_confid'] = "01/01/2014 - 01/31/2014"

#### save to CSV file
# Drop rows with no nt_group before saving
if 'nt_group' in combined.columns:
    before_count = len(combined)
    mask = combined['nt_group'].notna() & combined['nt_group'].astype(str).str.strip().ne('')
    combined = combined.loc[mask].reset_index(drop=True)
    after_count = len(combined)
    print(f"Dropped {before_count - after_count} rows with missing nt_group; {after_count} rows remain.")

combined.to_csv(home / 'Box' / 'Black_Lab' / 'projects' / 'TS' / 'New_Tics_R01' / 'Data' / 'analysis' / 'DS' / 'r01_combine' / f'r01_all_data_{current_date}.csv')

# Read csv file with columns for NT_r01_combine_for_ENIGMA
enigma_columns = pd.read_excel(home / 'Box' / 'Black_Lab' / 'projects' / 'TS' / 'New_Tics_R01' / 'Data' / 'analysis' / 'DS' / 'r01_combine' / 'NT_r01_combine_for_ENIGMA_columns.xlsx')
# filter for only the visit types required by ENIGMA
allowed_events = {
    'screening_visit_arm_1', 'scan_day_1_arm_1', 'repeat_scan_visit_arm_1',
    'initial_scan_visit_arm_1', 'initial_screen_ext_arm_1', 'initial_screen_extra_arm_1',
    '12_month_follow_up_arm_1', '12_month_scan_visi_arm_1'
}
if 'redcap_event_name' in combined.columns:
    combined_for_enigma = combined.loc[combined['redcap_event_name'].isin(allowed_events)].copy()
else:
    combined_for_enigma = combined.copy()

# Sort ENIGMA input by subject and visit date for consistent ordering
sort_cols = []
if 'demo_study_id' in combined_for_enigma.columns:
    sort_cols.append('demo_study_id')
if 'visit_date' in combined_for_enigma.columns:
    # convert visit_date to datetime for proper sorting; coerce errors to NaT
    combined_for_enigma['_enigma_visit_date_dt'] = pd.to_datetime(combined_for_enigma.get('visit_date'), errors='coerce')
    sort_cols.append('_enigma_visit_date_dt')
if sort_cols:
    combined_for_enigma = combined_for_enigma.sort_values(by=sort_cols).reset_index(drop=True)
    # drop the temporary datetime column if created
    if '_enigma_visit_date_dt' in combined_for_enigma.columns:
        combined_for_enigma.drop(columns=['_enigma_visit_date_dt'], inplace=True)

# Add `dci_date` for ENIGMA: set to `visit_date` only where `ts_dci_score_1` is numeric
if 'dci_date' not in combined_for_enigma.columns:
    if 'ts_dci_score_1' in combined_for_enigma.columns and 'visit_date' in combined_for_enigma.columns:
        is_num = pd.to_numeric(combined_for_enigma['ts_dci_score_1'], errors='coerce').notna()
        combined_for_enigma['dci_date'] = np.where(is_num, combined_for_enigma['visit_date'], pd.NaT)
    else:
        combined_for_enigma['dci_date'] = pd.NaT

# Ensure ksads5 fields are present in ENIGMA input (keep for lifetime logic)
for _col in ('ksads5_adhd_yn', 'ksads5_ocd_ce', 'ksads5_ocd_msp'):
    if _col not in combined_for_enigma.columns:
        combined_for_enigma[_col] = pd.NA

# Reorder ksads5 fields to appear immediately after their expert_diagnosis counterparts
_cols = list(combined_for_enigma.columns)
def _insert_after(base_col, insert_cols, cols_list):
    if base_col not in cols_list:
        return cols_list
    # remove insert_cols if already present
    cols_list = [c for c in cols_list if c not in insert_cols]
    base_idx = cols_list.index(base_col)
    for i, ic in enumerate(insert_cols, start=1):
        cols_list.insert(base_idx + i, ic)
    return cols_list

_cols = _insert_after('expert_diagnosis_adhd', ['ksads5_adhd_yn'], _cols)
_cols = _insert_after('expert_diagnosis_ocd', ['ksads5_ocd_ce', 'ksads5_ocd_msp'], _cols)
combined_for_enigma = combined_for_enigma.loc[:, _cols]

# Compute ADHD current flag on the ENIGMA input (used to derive lifetime at first visit)
if 'expert_diagnosis_adhd' in combined_for_enigma.columns:
    combined_for_enigma['ADHD_diagnosis_current'] = pd.NA
    _mask_adhd_present = pd.to_numeric(combined_for_enigma['expert_diagnosis_adhd'], errors='coerce').isin([1, 2, 3])
    if _mask_adhd_present.any():
        combined_for_enigma.loc[_mask_adhd_present, 'ADHD_diagnosis_current'] = (
            combined_for_enigma.loc[_mask_adhd_present, 'expert_diagnosis_adhd']
            .apply(lambda x: 'Y' if int(x) == 1 else 'N')
        )

# Vectorized ADHD lifetime: compute row-level assigned flags, then carry-forward 'Y' per subject
combined_for_enigma['ADHD_diagnosis_lifetime'] = pd.NA
if 'demo_study_id' in combined_for_enigma.columns:
    _expert_num = pd.to_numeric(combined_for_enigma.get('expert_diagnosis_adhd'), errors='coerce')
    _ksads_num = pd.to_numeric(combined_for_enigma.get('ksads5_adhd_yn'), errors='coerce')
    _expert_present = _expert_num.isin([1, 2, 3])
    _expert_y = _expert_num == 1
    _expert_n = _expert_present & (_expert_num != 1)
    _ksads_y = _ksads_num == 1
    _ksads_n = _ksads_num == 0
    assigned_y = (_expert_y) | (_ksads_y)
    assigned_n = (_expert_n) | (_ksads_n)
    # carry-forward any Y within each subject (subjects already sorted by visit_date)
    lifetime_y = assigned_y.groupby(combined_for_enigma['demo_study_id']).cummax()
    combined_for_enigma['ADHD_diagnosis_lifetime'] = np.where(lifetime_y, 'Y', np.where(assigned_n, 'N', pd.NA))
    # Set lifetime source to clinician when lifetime is determined (Y or N)
    combined_for_enigma['ADHD_dx_lifetime_source'] = pd.NA
    combined_for_enigma.loc[combined_for_enigma['ADHD_diagnosis_lifetime'].isin(['Y', 'N']), 'ADHD_dx_lifetime_source'] = 'clinician'

# For YGTSS variables: for screening-like events, keep YGTSS only for the latest
# screening visit per subject. Clear YGTSS values on other screening rows so ENIGMA
# receives only the most recent screening YGTSS measurements.
if 'demo_study_id' in combined_for_enigma.columns and 'redcap_event_name' in combined_for_enigma.columns:
    # identify ygtss columns (case-insensitive)
    ygtss_cols = [c for c in combined_for_enigma.columns if 'ygtss' in c.lower()]
    if ygtss_cols:
        # mark screening-like rows using explicit list of event names
        screen_events = {
            'screening_visit_arm_1', 'scan_day_1_arm_1', 'repeat_scan_visit_arm_1',
            'initial_scan_visit_arm_1', 'initial_screen_ext_arm_1'
        }
        screen_mask = combined_for_enigma['redcap_event_name'].astype(str).str.lower().isin(screen_events)
        # create a datetime column for ordering (visit_date may be missing or non-datetime)
        combined_for_enigma['_enigma_visit_date_dt'] = pd.to_datetime(combined_for_enigma.get('visit_date'), errors='coerce')
        # process per subject: find latest screening row by max visit_date and copy its YGTSS into the canonical screen row
        screening_rows = combined_for_enigma.loc[screen_mask]
        if not screening_rows.empty:
            latest_by_subject = {}

# Compute OCD current flag on the ENIGMA input (used to derive lifetime)
if 'expert_diagnosis_ocd' in combined_for_enigma.columns:
    combined_for_enigma['OCD_diagnosis_current'] = pd.NA
    _mask_ocd_present = pd.to_numeric(combined_for_enigma['expert_diagnosis_ocd'], errors='coerce').isin([1, 2, 3])
    if _mask_ocd_present.any():
        combined_for_enigma.loc[_mask_ocd_present, 'OCD_diagnosis_current'] = (
            combined_for_enigma.loc[_mask_ocd_present, 'expert_diagnosis_ocd']
            .apply(lambda x: 'Y' if int(x) == 1 else 'N')
        )

# Vectorized OCD lifetime: compute row-level assigned flags then carry-forward 'Y' per subject
combined_for_enigma['OCD_diagnosis_lifetime'] = pd.NA
if 'demo_study_id' in combined_for_enigma.columns:
    _expert_ocd = pd.to_numeric(combined_for_enigma.get('expert_diagnosis_ocd'), errors='coerce')
    _ksads_ce = pd.to_numeric(combined_for_enigma.get('ksads5_ocd_ce'), errors='coerce')
    _ksads_msp = pd.to_numeric(combined_for_enigma.get('ksads5_ocd_msp'), errors='coerce')
    _expert_present_ocd = _expert_ocd.isin([1, 2, 3])
    _expert_y_ocd = _expert_ocd == 1
    _expert_n_ocd = _expert_present_ocd & (_expert_ocd != 1)
    _ksads_y_ocd = (_ksads_ce.isin([3, 4])) | (_ksads_msp == 3)
    _ksads_n_ocd = (_ksads_ce == 0) | (_ksads_msp == 0)
    assigned_y_ocd = _expert_y_ocd | _ksads_y_ocd
    assigned_n_ocd = _expert_n_ocd | _ksads_n_ocd
    lifetime_y_ocd = assigned_y_ocd.groupby(combined_for_enigma['demo_study_id']).cummax()
    combined_for_enigma['OCD_diagnosis_lifetime'] = np.where(lifetime_y_ocd, 'Y', np.where(assigned_n_ocd, 'N', pd.NA))
    # Set lifetime source to clinician when lifetime is determined (Y or N)
    combined_for_enigma['OCD_dx_lifetime_source'] = pd.NA
    combined_for_enigma.loc[combined_for_enigma['OCD_diagnosis_lifetime'].isin(['Y', 'N']), 'OCD_dx_lifetime_source'] = 'clinician'

    target_indices = set()
    for sid, source_idx in latest_by_subject.items():
        # determine target row: prefer the canonical 'screening_visit_arm_1' row for this subject
        subj_mask = (combined_for_enigma['demo_study_id'] == sid)
        canonical_rows = combined_for_enigma.index[subj_mask & (combined_for_enigma['redcap_event_name'].astype(str).str.lower() == 'screening_visit_arm_1')].tolist()
        if canonical_rows:
            target_idx = canonical_rows[0]
        else:
            target_idx = source_idx

        # copy YGTSS columns and set ygtss_date on the target row
        for col in ygtss_cols:
            combined_for_enigma.loc[target_idx, col] = combined_for_enigma.loc[source_idx, col]
        combined_for_enigma.loc[target_idx, 'ygtss_date'] = combined_for_enigma.loc[source_idx, 'visit_date'] if 'visit_date' in combined_for_enigma.columns else pd.NaT
        target_indices.add(target_idx)

    # clear YGTSS on all screening-like rows that are NOT the target rows
    mask_to_clear = screen_mask & ~combined_for_enigma.index.isin(list(target_indices))
    combined_for_enigma.loc[mask_to_clear, ygtss_cols] = np.nan

    # PUTS: pick latest screening row that actually contains PUTS data (by date) and copy PUTS values
    # ensure puts_date column exists before use to avoid KeyError when reordering
    if 'puts_date' not in combined_for_enigma.columns:
        # ensure puts_date exists (start as missing; will be set from PUTS rows or from puts_total later)
        combined_for_enigma['puts_date'] = pd.NaT
    puts_cols = [c for c in combined_for_enigma.columns if c.lower().startswith('puts_')]
    if puts_cols:
        # find screening rows that have any PUTS data
        # screening_rows was created earlier and may not contain columns added to
        # combined_for_enigma afterward, so only use puts_cols that exist in screening_rows
        existing_puts_cols = [c for c in puts_cols if c in screening_rows.columns]
        if existing_puts_cols:
            screening_puts = screening_rows[screening_rows[existing_puts_cols].notna().any(axis=1)]
        else:
            screening_puts = screening_rows.iloc[0:0]
        latest_puts_by_subject = {}
        for sid, grp in screening_puts.groupby('demo_study_id', sort=False):
            if grp['_enigma_visit_date_dt'].notna().any():
                max_dt = grp['_enigma_visit_date_dt'].max()
                candidates = grp[grp['_enigma_visit_date_dt'] == max_dt]
                source_puts_idx = candidates.index[-1]
            else:
                source_puts_idx = grp.index[-1]
            latest_puts_by_subject[sid] = source_puts_idx

        for sid, source_idx in latest_by_subject.items():
            # determine which source to use for PUTS: prefer latest_puts_by_subject if available
            source_puts_idx = latest_puts_by_subject.get(sid, source_idx)
            subj_mask = (combined_for_enigma['demo_study_id'] == sid)
            canonical_rows = combined_for_enigma.index[subj_mask & (combined_for_enigma['redcap_event_name'].astype(str).str.lower() == 'screening_visit_arm_1')].tolist()
            if canonical_rows:
                target_idx = canonical_rows[0]
            else:
                target_idx = source_puts_idx
            for col in puts_cols:
                combined_for_enigma.loc[target_idx, col] = combined_for_enigma.loc[source_puts_idx, col]
            combined_for_enigma.loc[target_idx, 'puts_date'] = combined_for_enigma.loc[source_puts_idx, 'visit_date'] if 'visit_date' in combined_for_enigma.columns else pd.NaT
        # clear PUTS on non-target screening rows
        combined_for_enigma.loc[mask_to_clear, puts_cols] = np.nan

    # CYBOCS: copy latest screening row that has CYBOCS data (fields starting with 'cybocs_past_week_expert')
    cybocs_cols = [c for c in combined_for_enigma.columns if c.lower().startswith('cybocs_past_week_expert')]
    if cybocs_cols:
        # screening rows with any CYBOCS data
        screening_cybocs = screening_rows[screening_rows[cybocs_cols].notna().any(axis=1)]
        latest_cybocs_by_subject = {}
        for sid, grp in screening_cybocs.groupby('demo_study_id', sort=False):
            if grp['_enigma_visit_date_dt'].notna().any():
                max_dt = grp['_enigma_visit_date_dt'].max()
                candidates = grp[grp['_enigma_visit_date_dt'] == max_dt]
                source_cybocs_idx = candidates.index[-1]
            else:
                source_cybocs_idx = grp.index[-1]
            latest_cybocs_by_subject[sid] = source_cybocs_idx

        for sid, source_idx in latest_by_subject.items():
            # prefer latest with CYBOCS data if present, otherwise fallback to latest screening row
            source_cybocs_idx = latest_cybocs_by_subject.get(sid, source_idx)
            subj_mask = (combined_for_enigma['demo_study_id'] == sid)
            canonical_rows = combined_for_enigma.index[subj_mask & (combined_for_enigma['redcap_event_name'].astype(str).str.lower() == 'screening_visit_arm_1')].tolist()
            if canonical_rows:
                target_idx = canonical_rows[0]
            else:
                target_idx = source_cybocs_idx
            for col in cybocs_cols:
                combined_for_enigma.loc[target_idx, col] = combined_for_enigma.loc[source_cybocs_idx, col]
            # set cybocs_date on the target row from the source visit_date
            combined_for_enigma.loc[target_idx, 'cybocs_date'] = combined_for_enigma.loc[source_cybocs_idx, 'visit_date'] if 'visit_date' in combined_for_enigma.columns else pd.NaT
        # clear CYBOCS on non-target screening rows
        combined_for_enigma.loc[mask_to_clear, cybocs_cols] = np.nan

    # --- 12-month events: if the scan visit is later than the follow-up, move data into the canonical 12_month_follow_up_arm_1 row
    twelve_month_events = {'12_month_follow_up_arm_1', '12_month_scan_visi_arm_1'}
    mask_12mo = combined_for_enigma['redcap_event_name'].astype(str).str.lower().isin(twelve_month_events)
    twelve_rows = combined_for_enigma.loc[mask_12mo]
    if not twelve_rows.empty:
        puts_cols_12 = [c for c in combined_for_enigma.columns if c.lower().startswith('puts_')]
        cybocs_cols_12 = [c for c in combined_for_enigma.columns if c.lower().startswith('cybocs_past_week_expert')]
        adhd_source_field = 'adhd_current_expert_total'
        adhd_comment_field = 'adhd_current_expert_comments'
        adhd_cols_12 = [c for c in [adhd_source_field, adhd_comment_field] if c in combined_for_enigma.columns]
        for sid, grp in twelve_rows.groupby('demo_study_id', sort=False):
            # find canonical follow-up index
            follow_idx_list = grp[grp['redcap_event_name'].astype(str).str.lower() == '12_month_follow_up_arm_1'].index.tolist()
            if not follow_idx_list:
                continue
            follow_idx = follow_idx_list[0]
            # parse visit_date for this group's rows to robustly compare
            parsed_dates = pd.to_datetime(grp['visit_date'], errors='coerce')
            # choose a source row as the one with the max parsed date (fallback to last row)
            if parsed_dates.notna().any():
                max_dt = parsed_dates.max()
                candidates = grp[parsed_dates == max_dt]
                source_idx = candidates.index[-1]
            else:
                source_idx = grp.index[-1]

            follow_dt = pd.to_datetime(combined_for_enigma.loc[follow_idx, 'visit_date'], errors='coerce') if 'visit_date' in combined_for_enigma.columns else pd.NaT
            source_dt = pd.to_datetime(combined_for_enigma.loc[source_idx, 'visit_date'], errors='coerce') if 'visit_date' in combined_for_enigma.columns else pd.NaT

            # helper to check presence of data
            def _has_cols(idx, cols):
                return any(col in combined_for_enigma.columns and pd.notna(combined_for_enigma.loc[idx, col]) for col in cols)

            # For each measure, decide whether to copy from source->follow based on dates and presence
            # YGTSS
            has_source_ygtss = _has_cols(source_idx, ygtss_cols)
            has_follow_ygtss = _has_cols(follow_idx, ygtss_cols)
            should_copy_ygtss = False
            if not pd.isna(source_dt) and (pd.isna(follow_dt) or source_dt > follow_dt):
                should_copy_ygtss = has_source_ygtss
            elif not pd.isna(source_dt) and not pd.isna(follow_dt) and source_dt == follow_dt:
                should_copy_ygtss = has_source_ygtss and not has_follow_ygtss
            elif pd.isna(source_dt) and pd.isna(follow_dt):
                should_copy_ygtss = has_source_ygtss and not has_follow_ygtss

            if should_copy_ygtss:
                for col in ygtss_cols:
                    if col in combined_for_enigma.columns:
                        combined_for_enigma.loc[follow_idx, col] = combined_for_enigma.loc[source_idx, col]
                combined_for_enigma.loc[follow_idx, 'ygtss_date'] = combined_for_enigma.loc[source_idx, 'visit_date'] if 'visit_date' in combined_for_enigma.columns else pd.NaT

            # PUTS
            has_source_puts = _has_cols(source_idx, puts_cols_12)
            has_follow_puts = _has_cols(follow_idx, puts_cols_12)
            should_copy_puts = False
            if not pd.isna(source_dt) and (pd.isna(follow_dt) or source_dt > follow_dt):
                should_copy_puts = has_source_puts
            elif not pd.isna(source_dt) and not pd.isna(follow_dt) and source_dt == follow_dt:
                should_copy_puts = has_source_puts and not has_follow_puts
            elif pd.isna(source_dt) and pd.isna(follow_dt):
                should_copy_puts = has_source_puts and not has_follow_puts
            if should_copy_puts:
                for col in puts_cols_12:
                    combined_for_enigma.loc[follow_idx, col] = combined_for_enigma.loc[source_idx, col]
                combined_for_enigma.loc[follow_idx, 'puts_date'] = combined_for_enigma.loc[source_idx, 'visit_date'] if 'visit_date' in combined_for_enigma.columns else pd.NaT

            # CYBOCS
            has_source_cybocs = _has_cols(source_idx, cybocs_cols_12)
            has_follow_cybocs = _has_cols(follow_idx, cybocs_cols_12)
            should_copy_cybocs = False
            if not pd.isna(source_dt) and (pd.isna(follow_dt) or source_dt > follow_dt):
                should_copy_cybocs = has_source_cybocs
            elif not pd.isna(source_dt) and not pd.isna(follow_dt) and source_dt == follow_dt:
                should_copy_cybocs = has_source_cybocs and not has_follow_cybocs
            elif pd.isna(source_dt) and pd.isna(follow_dt):
                should_copy_cybocs = has_source_cybocs and not has_follow_cybocs
            if should_copy_cybocs:
                for col in cybocs_cols_12:
                    combined_for_enigma.loc[follow_idx, col] = combined_for_enigma.loc[source_idx, col]
                combined_for_enigma.loc[follow_idx, 'cybocs_date'] = combined_for_enigma.loc[source_idx, 'visit_date'] if 'visit_date' in combined_for_enigma.columns else pd.NaT

            # ADHD
            has_source_adhd = _has_cols(source_idx, adhd_cols_12)
            has_follow_adhd = _has_cols(follow_idx, adhd_cols_12)
            should_copy_adhd = False
            if not pd.isna(source_dt) and (pd.isna(follow_dt) or source_dt > follow_dt):
                should_copy_adhd = has_source_adhd
            elif not pd.isna(source_dt) and not pd.isna(follow_dt) and source_dt == follow_dt:
                should_copy_adhd = has_source_adhd and not has_follow_adhd
            elif pd.isna(source_dt) and pd.isna(follow_dt):
                should_copy_adhd = has_source_adhd and not has_follow_adhd
            if should_copy_adhd:
                for col in adhd_cols_12:
                    combined_for_enigma.loc[follow_idx, col] = combined_for_enigma.loc[source_idx, col]

            # clear these fields on other 12-month rows for this subject if we copied any
            if any([should_copy_ygtss, should_copy_puts, should_copy_cybocs, should_copy_adhd]):
                clear_mask = mask_12mo & (combined_for_enigma['demo_study_id'] == sid) & (combined_for_enigma.index != follow_idx)
                cols_to_clear = list(set(ygtss_cols + puts_cols_12 + cybocs_cols_12 + adhd_cols_12))
                combined_for_enigma.loc[clear_mask, cols_to_clear] = np.nan
            # Ensure date fields exist for follow-up if values are present but dates missing
            # (e.g., if data already on follow-up row but we never set *_date)
            visit_dt = combined_for_enigma.loc[follow_idx, 'visit_date'] if 'visit_date' in combined_for_enigma.columns else pd.NaT
            if 'ygtss_date' in combined_for_enigma.columns:
                # if ygtss exists but ygtss_date missing/empty, set to visit_date
                has_ygtss = combined_for_enigma.loc[follow_idx, ygtss_cols].notna().any() if ygtss_cols else False
                cur = combined_for_enigma.loc[follow_idx, 'ygtss_date']
                if has_ygtss and (pd.isna(cur) or str(cur).strip() == ''):
                    combined_for_enigma.loc[follow_idx, 'ygtss_date'] = visit_dt
            if 'puts_date' in combined_for_enigma.columns:
                has_puts = any(col in combined_for_enigma.columns and pd.notna(combined_for_enigma.loc[follow_idx, col]) for col in puts_cols_12)
                curp = combined_for_enigma.loc[follow_idx, 'puts_date']
                if has_puts and (pd.isna(curp) or str(curp).strip() == ''):
                    combined_for_enigma.loc[follow_idx, 'puts_date'] = visit_dt
            if 'cybocs_date' in combined_for_enigma.columns:
                has_cybocs = any(col in combined_for_enigma.columns and pd.notna(combined_for_enigma.loc[follow_idx, col]) for col in cybocs_cols_12)
                curc = combined_for_enigma.loc[follow_idx, 'cybocs_date']
                if has_cybocs and (pd.isna(curc) or str(curc).strip() == ''):
                    combined_for_enigma.loc[follow_idx, 'cybocs_date'] = visit_dt

    # ADHD: copy latest screening row that has adhd_current_expert_total data
    # add `adhd_sev_date` set to visit_date for ENIGMA processing (will be moved to latest row)
    if 'adhd_sev_date' not in combined_for_enigma.columns:
        if 'visit_date' in combined_for_enigma.columns:
            combined_for_enigma['adhd_sev_date'] = combined_for_enigma['visit_date']
        else:
            combined_for_enigma['adhd_sev_date'] = pd.NaT

    adhd_source_field = 'adhd_current_expert_total'
    adhd_comment_field = 'adhd_current_expert_comments'
    # include adhd_sev_date so it moves with the other ADHD fields
    adhd_cols = [c for c in [adhd_source_field, adhd_comment_field, 'adhd_sev_date'] if c in combined_for_enigma.columns]
    if adhd_cols:
        screening_adhd = screening_rows[screening_rows[adhd_source_field].notna()]
        latest_adhd_by_subject = {}
        for sid, grp in screening_adhd.groupby('demo_study_id', sort=False):
            if grp['_enigma_visit_date_dt'].notna().any():
                max_dt = grp['_enigma_visit_date_dt'].max()
                candidates = grp[grp['_enigma_visit_date_dt'] == max_dt]
                source_adhd_idx = candidates.index[-1]
            else:
                source_adhd_idx = grp.index[-1]
            latest_adhd_by_subject[sid] = source_adhd_idx

        for sid, source_idx in latest_by_subject.items():
            source_adhd_idx = latest_adhd_by_subject.get(sid, source_idx)
            subj_mask = (combined_for_enigma['demo_study_id'] == sid)
            canonical_rows = combined_for_enigma.index[subj_mask & (combined_for_enigma['redcap_event_name'].astype(str).str.lower() == 'screening_visit_arm_1')].tolist()
            if canonical_rows:
                target_idx = canonical_rows[0]
            else:
                target_idx = source_adhd_idx
            for col in adhd_cols:
                combined_for_enigma.loc[target_idx, col] = combined_for_enigma.loc[source_adhd_idx, col]
        # clear ADHD on non-target screening rows
        combined_for_enigma.loc[mask_to_clear, adhd_cols] = np.nan

        # drop temporary datetime helper
        combined_for_enigma.drop(columns=['_enigma_visit_date_dt'], inplace=True, errors='ignore')

# create `visit` column based on `redcap_event_name`
if 'redcap_event_name' in combined_for_enigma.columns:
    rn = combined_for_enigma['redcap_event_name'].astype(str).str.lower()
    is_screen = rn.str.contains('screen', na=False)
    is_12mo = rn.str.startswith('12_month', na=False)
    combined_for_enigma['visit'] = np.where(is_screen, 'ses-screen', np.where(is_12mo, 'ses-12mo', ''))
else:
    combined_for_enigma['visit'] = ''

# ensure `visit` is included immediately after `redcap_event_name` in the ENIGMA column ordering
enigma_cols = enigma_columns['r01_combine_name'].tolist()
if 'visit' not in enigma_cols:
    if 'redcap_event_name' in enigma_cols:
        idx = enigma_cols.index('redcap_event_name')
        enigma_cols.insert(idx+1, 'visit')
    else:
        # fallback: put visit at the front
        enigma_cols.insert(0, 'visit')

# Ensure dci_date appears immediately after ts_dci_score in enigma column ordering
if 'ts_dci_score' in enigma_cols and 'dci_date' not in enigma_cols:
    enigma_cols.insert(enigma_cols.index('ts_dci_score') + 1, 'dci_date')
elif 'dci_date' not in enigma_cols:
    enigma_cols.append('dci_date')

# add `kbit_date` to combined_for_enigma (only for screening visits)
if 'redcap_event_name' in combined_for_enigma.columns and 'visit_date' in combined_for_enigma.columns:
    combined_for_enigma['kbit_date'] = np.where(
        combined_for_enigma['redcap_event_name'] == 'screening_visit_arm_1',
        combined_for_enigma['visit_date'],
        pd.NaT
    )
else:
    combined_for_enigma['kbit_date'] = pd.NaT

# Ensure kbit_date appears immediately before kbit_verbal_knowledge_raw in enigma column ordering
if 'kbit_verbal_knowledge_raw' in enigma_cols and 'kbit_date' not in enigma_cols:
    idx = enigma_cols.index('kbit_verbal_knowledge_raw')
    enigma_cols.insert(idx, 'kbit_date')

# add `puts_date` set to visit_date only where puts_total > 0
if 'visit_date' in combined_for_enigma.columns and 'puts_total' in combined_for_enigma.columns:
    temp_puts_date = np.where(combined_for_enigma['puts_total'] > 0, combined_for_enigma['visit_date'], pd.NaT)
    # only set where puts_date is missing so earlier logic (copying latest PUTS) isn't overwritten
    if 'puts_date' in combined_for_enigma.columns:
        combined_for_enigma['puts_date'] = combined_for_enigma['puts_date'].fillna(pd.Series(temp_puts_date, index=combined_for_enigma.index))
    else:
        combined_for_enigma['puts_date'] = temp_puts_date
else:
    if 'puts_date' not in combined_for_enigma.columns:
        combined_for_enigma['puts_date'] = pd.NaT

# Ensure puts_date appears immediately before puts_1 in enigma column ordering
if 'puts_1' in enigma_cols and 'puts_date' not in enigma_cols:
    idx_puts = enigma_cols.index('puts_1')
    enigma_cols.insert(idx_puts, 'puts_date')

# Ensure adhd_sev_date appears immediately after adhd_current_expert_comments in enigma column ordering
if 'adhd_sev_date' not in combined_for_enigma.columns:
    if 'visit_date' in combined_for_enigma.columns:
        combined_for_enigma['adhd_sev_date'] = combined_for_enigma['visit_date']
    else:
        combined_for_enigma['adhd_sev_date'] = pd.NaT

if 'adhd_current_expert_comments' in enigma_cols and 'adhd_sev_date' not in enigma_cols:
    enigma_cols.insert(enigma_cols.index('adhd_current_expert_comments') + 1, 'adhd_sev_date')
elif 'adhd_sev_date' not in enigma_cols:
    enigma_cols.append('adhd_sev_date')

# Create `ancestry` by combining `race` and `ethnicity` for ENIGMA output
if 'race' in combined_for_enigma.columns or 'ethnicity' in combined_for_enigma.columns:
    def _combine_ancestry(row):
        r = row.get('race') if 'race' in row.index else None
        e = row.get('ethnicity') if 'ethnicity' in row.index else None
        if pd.isna(r) and pd.isna(e):
            return 'Unknown'
        parts = []
        if pd.notna(r) and str(r).strip() != '':
            parts.append(str(r))
        if pd.notna(e) and str(e).strip() != '':
            parts.append(str(e))
        return ', '.join(parts) if parts else 'Unknown'

    combined_for_enigma['ancestry'] = combined_for_enigma.apply(_combine_ancestry, axis=1)
    # ensure `ancestry` is in the enigma column ordering, right after `ethnicity` if possible
    if 'ancestry' not in enigma_cols:
        if 'ethnicity' in enigma_cols:
            enigma_cols.insert(enigma_cols.index('ethnicity') + 1, 'ancestry')
        else:
            enigma_cols.append('ancestry')

# Keep `race` and `ethnicity` in ENIGMA output (we also include `ancestry`)
# ensure `cybocs_date` column exists and is ordered after cybocs columns in ENIGMA output
if 'cybocs_date' not in combined_for_enigma.columns:
    combined_for_enigma['cybocs_date'] = pd.NaT
cybocs_enigma = [c for c in enigma_cols if c.startswith('cybocs_past_week_expert')]
if cybocs_enigma and 'cybocs_date' not in enigma_cols:
    last_c = cybocs_enigma[-1]
    enigma_cols.insert(enigma_cols.index(last_c) + 1, 'cybocs_date')

combined_enigma = combined_for_enigma.loc[:, [c for c in enigma_cols if c in combined_for_enigma.columns]]

# If tic_diagnosis_current is blank in the ENIGMA table, clear tic_dx_current_date as well
if 'tic_diagnosis_current' in combined_enigma.columns and 'tic_dx_current_date' in combined_enigma.columns:
    blank_dx_mask = combined_enigma['tic_diagnosis_current'].isna() | (combined_enigma['tic_diagnosis_current'].astype(str).str.strip() == '')
    combined_enigma.loc[blank_dx_mask, 'tic_dx_current_date'] = pd.NaT

# ADHD diagnosis fields: set current diagnosis flag and source
if 'expert_diagnosis_adhd' in combined_enigma.columns:
    # Initialize ADHD fields as empty; populate only where expert_diagnosis_adhd is present
    combined_enigma['ADHD_diagnosis_current'] = pd.NA
    combined_enigma['ADHD_dx_current_source'] = pd.NA
    combined_enigma['ADHD_dx_current_date'] = pd.NaT
    _mask_adhd = pd.to_numeric(combined_enigma['expert_diagnosis_adhd'], errors='coerce').isin([1, 2, 3])
    combined_enigma.loc[_mask_adhd, 'ADHD_diagnosis_current'] = combined_enigma.loc[_mask_adhd, 'expert_diagnosis_adhd'].apply(lambda x: 'Y' if x == 1 else 'N')
    combined_enigma.loc[_mask_adhd, 'ADHD_dx_current_source'] = 'clinician'
    if 'visit_date' in combined_enigma.columns:
        combined_enigma.loc[_mask_adhd, 'ADHD_dx_current_date'] = combined_enigma.loc[_mask_adhd, 'visit_date']
    else:
        combined_enigma.loc[_mask_adhd, 'ADHD_dx_current_date'] = pd.NaT

# OCD diagnosis fields: set current diagnosis flag and source
if 'expert_diagnosis_ocd' in combined_enigma.columns:
    # Initialize OCD fields as empty; populate only where expert_diagnosis_ocd is present
    combined_enigma['OCD_diagnosis_current'] = pd.NA
    combined_enigma['OCD_dx_current_source'] = pd.NA
    combined_enigma['OCD_dx_current_date'] = pd.NaT
    _mask_ocd = pd.to_numeric(combined_enigma['expert_diagnosis_ocd'], errors='coerce').isin([1, 2, 3])
    combined_enigma.loc[_mask_ocd, 'OCD_diagnosis_current'] = combined_enigma.loc[_mask_ocd, 'expert_diagnosis_ocd'].apply(lambda x: 'Y' if x == 1 else 'N')
    combined_enigma.loc[_mask_ocd, 'OCD_dx_current_source'] = 'clinician'
    if 'visit_date' in combined_enigma.columns:
        combined_enigma.loc[_mask_ocd, 'OCD_dx_current_date'] = combined_enigma.loc[_mask_ocd, 'visit_date']
    else:
        combined_enigma.loc[_mask_ocd, 'OCD_dx_current_date'] = pd.NaT

# Report how many ENIGMA columns are missing from combined
enigma_required = enigma_columns['r01_combine_name'].tolist()
missing_from_combined = [c for c in enigma_required if c not in combined_enigma.columns]
print(f"\nENIGMA columns required: {len(enigma_required)}; missing from combined: {len(missing_from_combined)}")
if missing_from_combined:
    print('Missing ENIGMA columns:')
    for c in missing_from_combined:
        print('-', c)


# Keep only canonical visit types for ENIGMA export: screening and 12-month follow-up
if 'redcap_event_name' in combined_enigma.columns:
    _keep_events = ['screening_visit_arm_1', '12_month_follow_up_arm_1']
    _before_count = len(combined_enigma)
    combined_enigma = combined_enigma.loc[combined_enigma['redcap_event_name'].isin(_keep_events)].copy()
    print(f"Filtered ENIGMA rows: kept {len(combined_enigma)} of {_before_count} rows (screening and 12-month follow-up).")
combined_enigma.to_csv(home / 'Box' / 'Black_Lab' / 'projects' / 'TS' / 'New_Tics_R01' / 'Data' / 'analysis' / 'DS' / 'r01_combine' / f'NT_r01_combine_for_ENIGMA_{current_date}.csv', index=False)