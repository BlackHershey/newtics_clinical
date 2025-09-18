import argparse
import pandas as pd
import csv

parser = argparse.ArgumentParser(description='Combines preR01 data with R01 data')
parser.add_argument('--preR01', help='preR01 input file', required=True)
parser.add_argument('--R01', help='R01 input file', required=True)
args = parser.parse_args()

nt = pd.read_csv(args.preR01)
r01 = pd.read_csv(args.R01)

#drops the columns in the input file
def drop_col(input_file):
    with open(input_file) as file:
        reader = csv.reader(file)
        for row in reader:
            allRows = [i for row in reader for i in row]
        delete_columns = [x for x in allRows if x]
    nt.drop(delete_columns, axis=1, inplace=True)
    
#takes in a list of tuples where the first element is combined with the second element
def combine_col(dic):
    old_drop = []
    for old_col, new_col in dic:
        nt[new_col] = nt.apply(lambda x: x[old_col] if pd.isna(x[new_col]) else x[new_col], axis=1)
        old_drop.append(old_col)
    nt.drop(old_drop, axis=1, inplace=True)
    
#3 columns
def combine_col_mvoa_3(dic):
    drop = []
    for col1, col2, new_col in dic:
        nt[new_col] = nt.apply(lambda x: 1 if (x[col1] == 1 or x[col2] == 1) else x[new_col], axis=1)
        drop.append(col1)
        drop.append(col2)
    nt.drop(drop, axis=1, inplace=True)

#2 columns
def combine_col_mvoa_2(dic):
    drop = []
    for col1, new_col in dic:
        nt[new_col] = nt.apply(lambda x: 1 if x[col1] == 1 else x[new_col], axis=1)
        drop.append(col1)
    nt.drop(drop, axis=1, inplace=True)
    
def combine_ksad(dic):
    ksad_delete = []
    for nt_name, r01_name in dic:
        ksad_delete.extend((f'ksads_{nt_name}_age_1ep', f'ksads_{nt_name}_dur_all_ep', f'ksads_{nt_name}_total_num_ep', f'ksads_{nt_name}_dur_ep'))
        rename = {f'ksads_{nt_name}_prev_ep': f'ksads5_{r01_name}_msp', f'ksads_{nt_name}_cur_ep': f'ksads5_{r01_name}_ce', f'ksads_{nt_name}_age_cur_ep': f'ksads5_{r01_name}_ce_ao'}
        nt.rename(columns=rename, inplace=True)
    nt.drop(ksad_delete, axis=1, errors='ignore', inplace=True)




#forms that were renamed from preR01 to R01
nt_rename_columns={'childs_age':'age_at_visit', 'incl_excl_braces': 'incl_excl_ortho', 'incl_excl_depression': 'incl_excl_disab_spec___7',
                 'incl_excl_autism': 'incl_excl_disab_spec___2', 'incl_excl_substance': 'incl_excl_disab_spec___3', 'childrens_yalebrown_oc_scale_cybocs_selfreport_sym_timestamp': 'childrens_yalebrown_oc_scale_self_report_timestamp', 'childrens_yalebrown_oc_scale_cybocs_selfreport_sym_complete': 'childrens_yalebrown_oc_scale_self_report_complete', 'cy_bocs_worst_ever_timestamp': 'cybocs_worst_ever_timestamp', 'cy_bocs_worst_ever_complete': 'cybocs_worst_ever_complete',
               'adhd_rating_scale_lifetime_parentself_timestamp': 'adhd_rating_scale_lifetime_timestamp', 'adhd_rating_scale_lifetime_parentself_complete': 'adhd_rating_scale_lifetime_complete', 'pedsql_pediatric_quality_of_life_timestamp': 'pediatric_quality_of_life_timestamp', 'pedsql_pediatric_quality_of_life_complete': 'pediatric_quality_of_life_complete', 'k_bit_2_complete': 'kbit_2_complete', 'puts_premonitory_urge_tics_scale_complete': 'premonitory_urge_tics_scale_complete', 'brief_neurological_exam_complete': 'expert_brief_neurological_exam_complete', 'ygtss_past_week_expert_complete': 'expert_ygtss_past_week_complete', 'expert_cybocs_symtoms_complete': 'expert_cybocs_symptoms_complete', 'cybocs_past_week_expert_complete': 'expert_cybocs_past_week_complete', 'adhd_rating_scale_current_expert_complete': 'expert_adhd_rating_scale_current_complete', 'ts_dci_scorable_complete': 'expert_ts_dci_complete',
               'ygtss_post_drz_complete': 'expert_ygtss_postdrz_complete', 'outcome_data_complete': 'expert_outcome_data_complete', 'ksads_diagnoses_complete': 'ksads_dsm5_diagnoses_complete', 'ksads_psych_disord': 'ksads5_psych_disord', 'ksads_mdd_diag_curr_ep': 'ksads_mdd_cur_ep', 'ksads_opd_specify': 'ksads5_othr1_dx_spec', 'ksads_opt': 'ksads5_outpthist_yn', 'ksads_opt_age': 'ksads5_outpthist_age', 'ksads_opt_total_dur': 'ksads5_outpthist_wks', 'ksads_psyhos': 'ksads5_psych_hosp_yn', 'ksads_psyhos_age': 'ksads5_psych_hosp_age', 'ksads_psyhos_num': 'ksads5_psych_hosp_x', 'ksads_ipt_total_dur': 'ksads5_psych_hosp_wks', 'ksads_suicidal_behavior___1': 'ksads5_suicide_ideation', 'ksads_suicidal_behavior___2': 'ksads5_suicide_gesture', 'ksads_suicidal_behavior___3': 'ksads5_suicide_attempt', 'ksads_reliability_of_info': 'ksads5_reliable_info', 'ksads_summary_lifetime_diagnoses_checklist_complete': 'ksads_pl_dsm5_summary_diagnoses_complete',
                  'drz_notes': 'drz_tics','fh_nd_sister_age_mvoa': 'fh_nd_sister_age', 'mafh_nd_sister_age_mvoa': 'mafh_nd_sister_age', 'mafh_nd_other_relative': 'mafh_nd_other_relative_age','mafh_rd_other_relative': 'mafh_rd_other_relative_age', 'mafh_th_other_relative': 'mafh_th_other_relative_age','mafh_5_other_relative': 'mafh_5_other_relative_age'}
nt.rename(columns=nt_rename_columns, inplace=True)

drop_col("C:\\Users\\songd\\Desktop\\David\\Black_lab\\data\\combine\\scripts\\general_preR01_delete.txt")
drop_col("C:\\Users\\songd\\Desktop\\David\\Black_lab\\data\\combine\\scripts\\ksads_preR01_delete.txt")
drop_col("C:\\Users\\songd\\Desktop\\David\\Black_lab\\data\\combine\\scripts\\mafh_preR01_delete.txt")
drop_col("C:\\Users\\songd\\Desktop\\David\\Black_lab\\data\\combine\\scripts\\fh_preR01_delete.txt")

mafh_combine = [('mafh_mom_age_preg', 'matern_age_mo'), ('mafh_dad_age_preg', 'matern_age_fa'), ('mafh_total_num_pregs', 'matern_no_preg'), ('mafh_total_num_livbir', 'matern_no_births'), ('mafh_threatened_abortion', 'matern_thrt_loss'),
               ('mafh_twin_pregnancy', 'matern_twins'), ('mafh_smoke', 'matern_cigs'), ('mafh_cigg', 'matern_cigs_no'), ('mafh_caf', 'matern_caff'), ('mafh_alc', 'matern_alcoh'), ('mafh_what_med_dur_prg___1', 'matern_preg_meds___1'), ('mafh_what_med_dur_prg___2', 'matern_preg_meds___2'),
               ('mafh_what_med_dur_prg___3', 'matern_preg_meds___3'), ('mafh_what_med_dur_prg___4', 'matern_preg_meds___4'), ('mafh_what_med_dur_prg___5', 'matern_preg_meds___5'), ('mafh_what_med_dur_prg___6', 'matern_preg_meds___9'), ('mafh_what_med_dur_prg___7', 'matern_preg_meds___10'), ('mafh_what_med_dur_prg___8', 'matern_preg_meds___11'),
               ('mafh_what_med_dur_prg___10', 'matern_preg_meds___14'), ('mafh_what_med_dur_prg___11', 'matern_preg_meds___15'), ('mafh_what_med_dur_prg___12', 'matern_preg_meds___16'), ('mafh_gestational_diabetes', 'matern_gest_diabetes'), ('mafh_abn_vag_bleeding', 'matern_vag_bleed'), ('mafh_if_yes_diag_given', 'matern_vag_bleed_expl'), ('mafh_hyperemesis', 'matern_nausea'),
               ('excessive_nausea', 'matern_nausea_expl'), ('mafh_pre_eclampsia', 'matern_pre_ecl'), ('mafh_maternal_malnutrition', 'matern_malnutri'), ('mafh_iugr', 'matern_slow_growth'), ('mafh_gestation_at_birth', 'matern_gest'), ('mafh_num_weeks_at_gestati', 'matern_gest_wks'), ('mafh_preterm_labor', 'matern_preterm_lbr'), ('mafh_prem_or_prolon_rup', 'matern_memb_rupt'), ('mafh_amnionitis', 'matern_infect_fld'), ('mafh_severe_pre_eclampsia', 'matern_sev_pclamps'),
               ('mafh_breech_presentation', 'matern_breech'), ('mafh_umb_cor_compress', 'matern_cord_comp'), ('mafh_fetal_distress', 'matern_fet_dist'), ('mafh_type_of_delivery', 'matern_del_type'), ('mafh_traumatic_delivery', 'matern_traum_del'), ('mafh_emerg_ces_section', 'matern_c_sec_emerg'), ('mafh_birth_before_admiss', 'matern_born_bef_hosp'), ('mafh_forceps_vacuum_used', 'matern_forc'), ('mafh_meconium_aspiration', 'matern_asp'), ('mafh_physical_injury', 'matern_lab_inj'),
               ('mafh_birth_weight_lbs_oz', 'matern_brth_wt'), ('mafh_was_your_child', 'matern_sz_gest_age'), ('mafh_hypothermia', 'matern_hypotherm'), ('mafh_hypoglycemia', 'matern_hypogly'), ('mafh_neonatal_jaundice', 'matern_jaund'), ('mafh_jaund_treat', 'matern_jaund_rx'), ('mafh_respiratory_distress', 'matern_breath_probs'), ('mafh_pneumonia', 'matern_pneum'), ('mafh_intracranial_bleed', 'matern_int_bleed'), ('mafh_necro_enterocolitis', 'matern_stom_probs'), ('mafh_nicu_admission', 'matern_nicu'), ('mafh_if_yes_dur_of_stay', 'matern_nicu_days'),
               ('mafh_ventilator', 'matern_vent'), ('mafh_father_age', 'fh_father_age_800'), ('mafh_mother_age', 'fh_father_age_800'), ('mafh_brother_age', 'fh_brother_age_800'), ('mafh_nd_brother_age', 'fh_nd_brother_age_800'), ('mafh_rd_brother_age', 'fh_rd_brother_age_800'), ('mafh_th_brother_age', 'fh_th_brother_age_800'), ('mafh_st_sister_age', 'fh_st_sister_age_800'), ('mafh_nd_sister_age', 'fh_nd_sister_age_mvoa_800'), ('mafh_rd_sister_age', 'fh_rd_sister_age_800'), ('mafh_th_sister_age', 'fh_th_sister_age_800'), ('mafh_st_other_relative_age', 'fh_st_other_relative_age_800'), ('mafh_nd_other_relative_age', 'fh_nd_other_relative_800'), ('mafh_rd_other_relative_age', 'fh_rd_other_relative_800'), ('mafh_th_other_relative_age', 'fh_th_other_relative_800'), ('mafh_5_other_relative_age', 'fh_5_other_relative_800')]              

combine_col(mafh_combine)


#generating list of mvoa family members
mvoa_3 = []
mvoa_2 = []
fam_list = ['father', 'mother', 'st_brother', 'nd_brother', 'rd_brother', 'th_brother', 'st_sister', 'nd_sister', 'rd_sister', 'th_sister', 'st_other', 'nd_other', 'rd_other', 'th_other', 'other_5']
for family in fam_list:
    for x in range(5):
        if x == 0:
            mvoa_2.append((f'fh_{family}_mvoa___{x}', f'fh_{family}_mvoa___{x}'))
        else:
            mvoa_3.append((f'mafh_{family}_mvoa___{x}', f'fh_{family}_mvoa___{x}', f'fh_{family}_mvoa_800___{x}'))
            
combine_col_mvoa_2(mvoa_2)
combine_col_mvoa_3(mvoa_3)

ts_dci_score_map = {'ts_dci_score_1': 15, 'ts_dci_score_2': 5, 'ts_dci_score_3': 5, 'ts_dci_score_4': 5, 'ts_dci_score_5': 7, 'ts_dci_score_6': 12, 'ts_dci_score_7': 4, 
                   'ts_dci_score_8': 7, 'ts_dci_score_9': 1, 'ts_dci_score_10': 2, 'ts_dci_score_11': 2, 'ts_dci_score_12': 2, 'ts_dci_score_13': 4, 'ts_dci_score_14': 1,
                   'ts_dci_score_15': 1, 'ts_dci_score_16': 1, 'ts_dci_score_17': 2, 'ts_dci_score_18': 4, 'ts_dci_score_19': 1, 'ts_dci_score_20': 2, 'ts_dci_score_21': 2,
                   'ts_dci_score_22': 4, 'ts_dci_score_23': 2, 'ts_dci_score_24': 2, 'ts_dci_score_25': 2, 'ts_dci_score_26': 4, 'ts_dci_score_27': 1} 

#converts absent from 1 to 0
ts_dci = []
ts_dci_score = []
for x in range(1, 28):
    ts_dci.append(f'ts_dci_{x}')
    ts_dci_score.append(f'ts_dci_score_{x}')
for dci in ts_dci:
    nt[dci].replace({1:0}, inplace=True)
ts_dci_dic = list(map(lambda x, y:(x,y), ts_dci, ts_dci_score))
combine_col(ts_dci_dic)
for dci in ts_dci_score:
    nt[dci] = nt.apply(lambda x: ts_dci_score_map[dci] if not x[dci] == 0 and not pd.isna(x[dci]) else x[dci], axis=1)
    
#srs_code
srs_dic = {'srs_q1': [0, 1, 2, 3, 'mo'], 'srs_q2': [0, 1, 2, 3, 'a'], 'srs_q3': [3, 2, 1, 0, 'mo'], 'srs_q4': [0, 1, 2, 3, 'ma'], 'srs_q5': [0, 1, 2, 3, 'cog'], 'srs_q6': [0, 1, 2, 3, 'mo'], 'srs_q7': [3, 2, 1, 0, 'a'], 'srs_q8': [0, 1, 2, 3, 'ma'], 'srs_q9': [0, 1, 2, 3, 'mo'], 'srs_q10': [0, 1, 2, 3, 'cog'], 'srs_q11': [3, 2, 1, 0, 'mo'], 'srs_q12': [3, 2, 1, 0, 'com'],
          'srs_q13': [0, 1, 2, 3, 'com'], 'srs_q14': [0, 1, 2, 3, 'ma'], 'srs_q15': [3, 2, 1, 0, 'cog'], 'srs_q16': [0, 1, 2, 3, 'com'], 'srs_q17': [3, 2, 1, 0, 'cog'], 'srs_q18': [0, 1, 2, 3, 'com'], 'srs_q19': [0, 1, 2, 3, 'com'], 'srs_q20': [0, 1, 2, 3, 'ma'], 'srs_q21': [3, 2, 1, 0, 'com'], 'srs_q22': [3, 2, 1, 0, 'com'], 'srs_q23': [0, 1, 2, 3, 'mo'], 'srs_q24': [0, 1, 2, 3, 'ma'],
           'srs_q25': [0, 1, 2, 3, 'a'], 'srs_q26': [3, 2, 1, 0, 'com'], 'srs_q27': [0, 1, 2, 3, 'mo'], 'srs_q28': [0, 1, 2, 3, 'ma'], 'srs_q29': [0, 1, 2, 3, 'ma'], 'srs_q30': [0, 1, 2, 3, 'cog'], 'srs_q31': [0, 1, 2, 3, 'man'], 'srs_q32': [3, 2, 1, 0, 'a'], 'srs_q33': [0, 1, 2, 3, 'com'], 'srs_q34': [0, 1, 2, 3, 'mo'], 'srs_q35': [0, 1, 2, 3, 'com'], 'srs_q36': [0, 1, 2, 3, 'com'], 'srs_q37': [0, 1, 2, 3, 'com'],
           'srs_q38': [3, 2, 1, 0, 'com'], 'srs_q39': [0, 1, 2, 3, 'ma'], 'srs_q40': [3, 2, 1, 0, 'cog'], 'srs_q41': [0, 1, 2, 3, 'com'], 'srs_q42': [0, 1, 2, 3, 'cog'], 'srs_q43': [3, 2, 1, 0, 'mo'], 'srs_q44': [0, 1, 2, 3, 'cog'], 'srs_q45': [3, 2, 1, 0, 'a'], 'srs_q46': [0, 1, 2, 3, 'com'], 'srs_q47': [0, 1, 2, 3, 'com'], 'srs_q48': [3, 2, 1, 0, 'cog'], 'srs_q49': [0, 1, 2, 3, 'ma'], 
           'srs_q50': [0, 1, 2, 3, 'ma'], 'srs_q51': [0, 1, 2, 3, 'com'], 'srs_q52': [3, 2, 1, 0,'a'], 'srs_q53': [0, 1, 2, 3, 'com'], 'srs_q54': [0, 1, 2, 3,'a'], 'srs_q55': [3, 2, 1, 0, 'com'], 'srs_q56': [0, 1, 2, 3, 'a'], 'srs_q57': [0, 1, 2, 3, 'com'], 'srs_q58': [0, 1, 2, 3, 'cog'], 'srs_q59': [0, 1, 2, 3, 'cog'], 'srs_q60': [0, 1, 2, 3, 'com'], 'srs_q61': [0, 1, 2, 3, 'com'], 'srs_q62': [0, 1, 2, 3, 'cog'], 
           'srs_q63': [0, 1, 2, 3, 'ma'], 'srs_q64': [0, 1, 2, 3, 'mo'], 'srs_q65': [0, 1, 2, 3, 'mo']
}

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

#generates srs list
srs_list = []
srs_list_800 = []
for x in range(1,66):
    srs_list.append(f'srs_q{x}')
    srs_list_800.append(f'srs_800_q{x}')
for srs in srs_list:
    nt[srs] = nt.apply(lambda x: srs_dic[srs][int(x[srs])-1] if not pd.isna(x[srs]) else x[srs], axis=1)
srs_combine = list(map(lambda x, y:(x,y), srs_list, srs_list_800))
combine_col(srs_combine)

awareness = []
cognition = []
communication = []
motivation = []
mannerisms = []
for key, value in srs_dic.items():
    if value[4] == 'a':
        awareness.append(f'srs_800_{key[4:]}')
    elif value[4] == 'cog':
        cognition.append(f'srs_800_{key[4:]}')
    elif value[4] == 'com':
        communication.append(f'srs_800_{key[4:]}')
    elif value[4] == 'mo':
        motivation.append(f'srs_800_{key[4:]}')
    elif value[4] == 'ma':
        mannerisms.append(f'srs_800_{key[4:]}')
        
nt['srs_awareness'] = nt[awareness].sum(axis=1, min_count=1)
nt['srs_cognition'] = nt[cognition].sum(axis=1, min_count=1)
nt['srs_communication'] = nt[communication].sum(axis=1, min_count=1)
nt['srs_motivation'] = nt[motivation].sum(axis=1, min_count=1)
nt['srs_mannerisms'] = nt[mannerisms].sum(axis=1, min_count=1)


r01_rename_columns={'ksads5_provisioaltic_msp': 'ksads5_provisionaltic_msp', 'ksads5_provisioaltic_msp_ao': 'ksads5_provisionaltic_msp_ao'}
r01.rename(columns=r01_rename_columns, inplace=True)


ksad_conversion = [("dysth", "dysth"), ("addm", "mddaddm"), ("pd", "panic"), ("sad", "sepanxiety"), ("agor", "agoraphobia"), ("gad", "gad"),
                  ("ocd", "ocd"), ("adam", "adjdo_anx"), ("enur", "enuresis"), ("enco", "encopresis"), ("an", "anorexia"), ("bul", "bulimia"), ("cd", "cd"),
                  ("odd", "odd"), ("addc", "adjdo_doc"), ("admmc", "adjdo_mix"), ("ts", "tourettes"), ("cmvtd", "ctd"), ("ttd", "provisionaltic"), ("mdd", "mdd"),
                  ("opd", "othr1_dx")]
combine_ksad(ksad_conversion)

r01_columns = frozenset(r01.columns)
combined_variables = [x for x in nt.columns if x in r01_columns]


combined = pd.concat([nt, r01], axis=0).sort_values(by=['demo_study_id']).reset_index(drop=True)

#calculating SRS t scores
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

#male tscore dic
male_total_tscore = {}
counter = 34
skips = [16, 39, 62, 87, 110]
for x in range (0, 117):
    male_total_tscore[x] = int(counter)
    if x not in skips: 
        counter += 0.5

#female tscore dic
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
        
nt_columns = frozenset(combined_variables)
discrepancies = [x for x in nt.columns if x not in nt_columns]
print("Number of discrepancies: ", len(discrepancies))
print(discrepancies)

combined.to_csv('r01_all_data.csv')