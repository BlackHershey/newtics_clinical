import numpy as np
import pandas as pd
import csv
import datefinder
import json
import os
import pyodbc
import re

import sys
sys.path.append('..')
import common

from bs4 import BeautifulSoup
from datetime import datetime
from getpass import getpass, getuser
from gooey import Gooey, GooeyParser
from itertools import product
from os.path import join
from redcap import Project, RedcapError

EVENT_NAME = 'event_name'
NTID = 'demo_study_id'

RESULT_OUTPUT_FILE = '{}_result.csv'

EXCLUDED = [ 'test' ] + [ 'NT' + str(sub) for sub in [706, 708, 711, 717, 721, 723, 725, 727, 737, 739, 802, 803, 804, 784] ]
ANXIETY_DISORDERS = [ ('pd', 'panic', 'Panic Disorder'), ('sad', 'sepanxiety', 'Separation Anxiety Disorder'),
	('adc', None, 'Avoidant Disorder of Childhood'), ('sp', 'specphobia', 'Simple Phobia'), ('socp', None, 'Social Phobia'),
	('agor', 'agoraphobia', 'Agoraphobia'), ('oad', None, 'Overanxious Disorder'), ('gad', None, 'Generalized Anxiety Disorder') ]


ALPHA_ONLY = re.compile('([^\sa-zA-Z]|_)+')

def rename_events(df, study_vars):
	# rename columns to be more friendly for flattened (all events on one row) display
	df.rename(columns={'redcap_event_name': EVENT_NAME}, inplace=True)
	df[EVENT_NAME] = df[EVENT_NAME].apply(lambda n: n.replace('_arm_1', ''))
	df[EVENT_NAME] = df[EVENT_NAME].replace(list(study_vars['event_name_renames'].keys()), list(study_vars['event_name_renames'].values()))
	return df.set_index([NTID, EVENT_NAME])


def get_dataframe_from_file(file_name, study_vars):
	path = os.path.join(os.getcwd(), file_name)
	df = pd.read_csv(path)
	return rename_events(df, study_vars)


def get_dataframe_from_api(study_name, study_vars, db_password, fields=None):
	project = common.get_redcap_project(study_name, db_password)
	df = project.export_records(format='df', fields=fields)
	return rename_events(df.reset_index(), study_vars)


def get_matching_columns(columns, pattern):
	return [ col for col in columns if re.search(pattern, col) ]


# helper function to sort event names by actual number (12, 24, etc.) instead of string order
#	if no number in event name (aka 'screen'), assign it to 0 to make it come first
# 	-- for handling 'ever_had' that spans multiple sessions (need to ensure chronological order of events)
def event_sort(column):
	event_name = column.rsplit('_', 1)[1]
	return [ int(s) if s.isdigit() else 0 for s in re.split('(\d+)', event_name) ]


def multi_session_ever_had(df, characteristic):
	cols = [ col for col in df.columns if characteristic in col ]
	cols.sort(key=event_sort)

	for i in range(len(cols)):
		visit = cols[i].rsplit('_', 1)[1]
		visit_col = '_'.join([characteristic, visit])
		ever_had_col = '_'.join([characteristic, 'ever', visit])
		df[ever_had_col] = df[cols[:i+1]].apply(lambda x: any(val == True for val in x.values) if not pd.isnull(x[visit_col]) else None, axis=1)
	return df


def ever_had(characteristic, df, columns):
	if all(col in df.columns for col in columns):
		df[characteristic] = df[columns].apply(lambda x: any(val == 'Y' for val in x.values) if not pd.isnull(x).all() else None, axis=1)
	else:
		df[characteristic] = np.nan
	return df

def calc_scores_for_columns(df, column_prefixes, suffix='total'):
	scored_columns = []
	for i in range(len(column_prefixes)):
		# find columns that fit the pattern
		pattern = '^' + column_prefixes[i]
		pattern_matched_columns = get_matching_columns(df.columns, pattern)

		scored_column = '_'.join([column_prefixes[i], suffix])
		# add new column to dataframe for the score of the matched columns
		if suffix == 'total':
			scored_columns.append(scored_column)
			df[scored_column] = df[pattern_matched_columns].sum(axis=1, min_count=1)
		elif suffix == 'avg':
			scored_columns.append(scored_column)
			# do not score if fewer than 50% of questions have been answered
			df[scored_column] = df[pattern_matched_columns].apply(lambda x: x.mean() if (x.isna().sum() < len(x) / 2) else np.nan, axis=1)

	return df[scored_columns] # retain only the scored columns


def score_pedqol(pedqol_df):
	pedqol_df = pedqol_df.replace([1, 2, 3, 4, 5], [ 100, 75, 50, 25, 0])
	subscales = ['phys', 'emo', 'social', 'school']
	col_patterns = [ 'pedsql_' + col for col in subscales ]
	pedqol_scores = calc_scores_for_columns(pedqol_df, col_patterns, suffix='avg')
	pedqol_scores['pedsql_physical_summary_score'] = pedqol_scores[col_patterns[0] + '_avg']
	pedqol_scores['pedsql_pyschosocial_summary_score'] = pedqol_scores[[ col + '_avg' for col in col_patterns[1:] ]].sum(axis=1)
	return pedqol_scores


def score_ses(ses_df):
	edu_level_columns = get_matching_columns(ses_df, 'ses_edu_level')
	occ_columns = get_matching_columns(ses_df, 'ses_occ')
	ses_df['ses_avg'] = (ses_df[edu_level_columns].mean(axis=1)).add(ses_df[occ_columns].mean(axis=1))
	return ses_df['ses_avg']


def score_adhd(adhd_df):
	return calc_scores_for_columns(adhd_df, ['adhd_lifetime_self', 'adhd_current_expert', 'adhd_current_parent'])


def score_cybocs(cybocs_df):
	return calc_scores_for_columns(cybocs_df, ['cybocs_worst_ev', 'cybocs_past_week_expert', 'cybocs_lifetime_parent'])


def score_puts(puts_df):
	missing_vals = puts_df.isnull().sum(axis=1)
	puts_df = calc_scores_for_columns(puts_df, ['puts'])
	puts_df += 9
	puts_df['puts_total_completed_only'] = puts_df['puts_total'][missing_vals == 0]
	return puts_df


def apply_dci_scoring_strategy(row, columns, scorable_columns, multiplier_map):
	suppress_col = 'dci_attempts_to_suppress_tics'

	scorable_complete_cols = [ col for col in columns if re.search('\w*ts_dci\w*complete', col) ]
	if pd.notnull(row[scorable_columns]).any():
		row[suppress_col] = row['ts_dci_score_16'] == 1
		row['dci_total'] = row[scorable_columns].sum() # for newer participants multipliers are built-in - can simply sum responses
	elif pd.notnull(row[list(multiplier_map.keys())]).any():
		row[suppress_col] = row['ts_dci_16'] == 2
		total = 0
		for key in multiplier_map:
			response = row[key]
			if pd.notnull(response):
				total += (response-1) * multiplier_map[key] # subtract 1 from response since response marked 1 for absent, 2 for present
		row['dci_total'] = total
	else:
		row['dci_total'] = np.nan
		row[suppress_col] = np.nan

	return row[['dci_total', suppress_col]]


def score_dci(dci_df, study_vars):
	scorable_columns = get_matching_columns(dci_df, '^ts_dci_score_\d+$')

	# generate mapping of question to multiplier for scoring (for participants who joined prior to scorable form creation)
	multipliers = [15, 5, 5, 5, 7, 12, 4, 7, 1, 2, 2, 2, 4, 1, 1, 1, 2, 4, 1, 2, 2, 4, 2, 2, 2, 4, 1]
	multiplier_mapping = {}
	for i in range(len(multipliers)):
		multiplier_mapping['ts_dci_' + str(i+1)] = multipliers[i]

	dci_df = dci_df.apply(apply_dci_scoring_strategy, args=(dci_df.columns, scorable_columns, multiplier_mapping), axis=1)
	return dci_df


def score_ygtss(ygtss_df):
	impairment_score_cols = get_matching_columns(ygtss_df.columns, 'ygtss.+_p_6') # both past_week_expert and post_drz

	# store impairment score locally before removing from dataframe
	# needs to be removed as it would affect the summing of phonetic tics
	impairment_scores = ygtss_df[impairment_score_cols]
	ygtss_df = ygtss_df.drop(impairment_score_cols, axis=1)

	# get the scores for both motor and phonetic tics individually
	tic_severity_columns = ['ygtss_past_week_expert_m', 'ygtss_past_week_expert_p', 'ygtss_post_drz_m', 'ygtss_post_drz_p']
	ygtss_df = calc_scores_for_columns(ygtss_df, tic_severity_columns)

	ygtss_df['ygtss_post_drz_total_tic'] = ygtss_df[[ col + '_total' for col in tic_severity_columns[2:]]].sum(axis=1, min_count=1) # sum post drz motor and phonetic tic scores
	ygtss_df['ygtss_past_week_expert_total_tic'] = ygtss_df[[ col + '_total' for col in tic_severity_columns[:2]]].sum(axis=1, min_count=1) # sum past week motor and phonetic tic scores
	ygtss_df[['ygtss_past_week_expert_total_impairment', 'ygtss_post_drz_total_impairment']] = impairment_scores # add impairment scores back into dataframe
	ygtss_df['ygtss_minimal_impairment'] = ygtss_df['ygtss_past_week_expert_total_impairment'].apply(lambda x: x <= 10 if not pd.isnull(x) else None)
	return ygtss_df

def calculate_handedness(row, last_question=10):
	row.replace(0, np.nan, inplace = True)
	if row.isnull().all():
		row['handedness'] = None
		row['dominant_hand'] = None
	else:
		numerator_sum = 0
		denominator_sum = 0
		for i in range(1, last_question+1):
			column = 'edinburgh_handedness_' + str(i)
			if abs(row[column]) < 3:
				numerator_sum += row[column]
				denominator_sum += abs(row[column])
			elif abs(row[column]) == 4:
				denominator_sum += 2
			# else value is treated as 0

		row['edinburgh_handedness'] = float(numerator_sum) / denominator_sum
		if row['edinburgh_handedness'] > 0:
			row['dominant_hand'] = 'R'
		elif row['edinburgh_handedness'] < 0:
			row['dominant_hand'] = 'L'
		else:
			row['dominant_hand'] = 'N' # how do we handle 0?

	return row


def score_edinburgh(edinburgh_df):
	edinburgh_df = edinburgh_df.apply(calculate_handedness, axis=1)
	return edinburgh_df[['edinburgh_handedness', 'dominant_hand']]


def map_peg_by_handedness(row):
	if row['dominant_hand'] == 'R':
		row['peg_dominant_30s'] = row['peg_right_30s']
		row['peg_nondominant_30s'] = row['peg_left_30s']
	else:
		row['peg_dominant_30s'] = row['peg_left_30s']
		row['peg_nondominant_30s'] = row['peg_right_30s']
	return row


def score_pegboard(peg_df):
	peg_df = peg_df.apply(map_peg_by_handedness, axis=1)
	return peg_df[['peg_dominant_30s', 'peg_nondominant_30s', 'peg_both_30s', 'peg_assembly_60s']]


def score_expert_rated_diagnoses(expert_dx_df):
	measures = ['tourette', 'chronic_tics', 'transient', 'simple_motor', 'simple_vocal', 'complex_motor', 'complex_vocal', 'adhd', 'ocd']
	columns = [ '_'.join(['expert_diagnosis', measure]) for measure in measures ]
	expert_dx_df[columns] = expert_dx_df[columns].replace([1, 2, 3, 4], ['Y', 'N', 'N', 'N']) # only consider 'present' as diagnosed
	expert_dx_df = ever_had('vocal_tics_ever', expert_dx_df, ['expert_diagnosis_simple_vocal', 'expert_diagnosis_complex_vocal'])
	expert_dx_df = ever_had('complex_tics_ever', expert_dx_df, ['expert_diagnosis_complex_motor', 'expert_diagnosis_complex_vocal'])
	expert_dx_df['expert_diagnosis_awareness'] = expert_dx_df['expert_diagnosis_awareness'].apply(lambda x: x.lower() == 'y' or 'yes' in x.lower() if pd.notnull(x) else np.nan)
	return expert_dx_df[columns + ['vocal_tics_ever', 'complex_tics_ever', 'expert_diagnosis_awareness']]


def get_ksads_diagnoses(row, data_dict):
	diagnosed_vars = row.index[row == 'Y'].values
	diagnoses = []
	for var in diagnosed_vars:
		label = ALPHA_ONLY.sub('', BeautifulSoup(data_dict[data_dict['Variable / Field Name'] == var].iloc[0]['Field Label'], 'lxml')
			.get_text()).rsplit(' ', 2)[0].replace('Diagnosis', '').strip() # find label in data dict, strip out html tags/other special character, get rid of episode string
		if var.endswith('prev_ep') or var.endswith('msp'):
			label += ' (Past)' # add back past designation for previous episode (otherwise, current episode is assumed)
		diagnoses.append(label)
	return '; '.join(diagnoses)


def score_ksads(ksads_df):
	prev_cur_ep_columns = get_matching_columns(ksads_df.columns, 'ksads_([a-z]+)_(prev|cur)_ep$')
	ce_msp_columns = get_matching_columns(ksads_df.columns, 'ksads5_([a-z]+_)+(ce|msp)$')
	all_episode_columns = prev_cur_ep_columns + ce_msp_columns
	ksads_df[ce_msp_columns] = ksads_df[ce_msp_columns] + 1 # newer form is 0-4 (3 and 4 techincally should be switched, but both count as diagnosed so it doesn't matter)
	ksads_df[all_episode_columns] = ksads_df[all_episode_columns].replace([np.nan, 1, 2, 3, 4, 5], ['N', 'N', 'N', 'N', 'Y', 'Y']) # consider 'definite' and 'partial remission' as diagnosed

	adhd_episode_cols = [ col for col in all_episode_columns if re.search('_(add|adhd)_', col) ]
	ocd_episode_cols = [ col for col in all_episode_columns if re.search('_ocd_', col) ]
	anx_search_options = '|'.join([ d for d in sum([tup[:-1] for tup in ANXIETY_DISORDERS ], ()) if d is not None ]) # get flattend list of both form versions disorder names (https://stackoverflow.com/a/10636583)
	anxiety_episode_cols = [ col for col in all_episode_columns if re.search('_(' + anx_search_options + '_)', col) ]
	named_disorder_columns = adhd_episode_cols + ocd_episode_cols + anxiety_episode_cols

	data_dict_df = pd.concat([
		pd.read_csv(r'C:\Users\{}\Box\Black_Lab\projects\TS\NewTics\Data\REDCap\NewTics_DataDictionary_2016-03-08.csv'.format(getuser())),
		pd.read_csv(r'C:\Users\{}\Box\Black_Lab\projects\TS\New Tics R01\Data\REDCap\REDCap_data_dictionaries\NewTicsR01_DataDictionary_2018-11-09.csv'.format(getuser()))
	])
	ksads_df['ksads_all_diagnoses'] = ksads_df[all_episode_columns].apply(get_ksads_diagnoses, args=(data_dict_df,), axis=1)
	ksads_df['ksads_anxiety_diagnoses'] = ksads_df[anxiety_episode_cols].apply(get_ksads_diagnoses, args=(data_dict_df,), axis=1)

	tic_columns = get_matching_columns(all_episode_columns, 'ksads\w*_(ts|cmvtd|ttd)_\w*')
	ksads_df = ever_had('adhd_ever', ksads_df, adhd_episode_cols)
	ksads_df = ever_had('ocd_ever', ksads_df, ocd_episode_cols)
	ksads_df = ever_had('other_anxiety_disorder_ever', ksads_df, anxiety_episode_cols)
	ksads_df = ever_had('other_ksads_ever', ksads_df, [ col for col in all_episode_columns if col not in named_disorder_columns ])
	ksads_df = ever_had('non_tic_ksads_ever', ksads_df, [ col for col in all_episode_columns if col not in tic_columns ])
	return ksads_df[['adhd_ever', 'ocd_ever', 'other_anxiety_disorder_ever', 'other_ksads_ever', 'non_tic_ksads_ever', 'ksads_all_diagnoses', 'ksads_anxiety_diagnoses']]


def parse_data_dictionary(file):
	response_mappings = []
	with open(file, encoding='utf-8-sig') as f:
		reader = csv.reader(f)
		for row in reader:
			if '_q' in row[0]:
				options = row[5].split('| ')
				mappings = [ option.replace(',', '').split(' ')[:2] for option in options ]
				response_mappings.append({ int(mapping[1]): int(mapping[0]) for mapping in mappings })
	return response_mappings


def score_srs(srs_df, sex_df):
	dd = r'C:\Users\{}\Box\Black_Lab\projects\TS\NewTics\Data\REDCap\SRS_800_datadictionary.csv'.format(getuser())
	response_mappings = parse_data_dictionary(dd)

	srs_700_columns = get_matching_columns(srs_df.columns, 'srs_q\d{1,2}')

	for idx, col in enumerate(srs_700_columns):
		srs_df[col].replace(response_mappings[idx], inplace=True)

	srs_total_df = calc_scores_for_columns(srs_df, ['srs_800_q', 'srs_q'])
	srs_df['srs_raw_score'] = srs_total_df.apply(lambda x: x['srs_800_q_total'] if pd.isnull(x['srs_q_total']) else x['srs_q_total'], axis=1)

	srs_df = srs_df.join(sex_df, how='left')
	srs_df['srs_tscore'] = srs_df.apply(lambda x: x['srs_raw_score']*.5523 + 34.7612 if x['sex'] == 'F' else x['srs_raw_score']*.4786 + 33.8637, axis=1)
	return srs_df['srs_tscore'].round()


def score_demographics(row):
	if row.isnull().all():
		row['non_white'] = np.nan
		row['hispanic'] = np.nan
	else:
		if row['demo_race___7'] == 1:
			row['non_white'] = np.nan
		elif row['demo_race___5'] == 1:
			row['non_white'] = False
		else:
			race_columns = [ 'demo_race___' + str(i) for i in range(1, 5) ]
			row['non_white'] = row[race_columns].sum() > 0

		if row['demo_ethnicity___1'] == 1:
			row['hispanic'] = False
		elif row['demo_ethnicity___2'] == 1:
			row['hispanic'] = True
		else:
			row['hispanic'] = np.nan

	return row[['sex', 'non_white', 'hispanic']]

def apply_extra_screen_data(row, ext_cols):
	ext_cols_not_null = [ col for col in ext_cols if not pd.isnull(row[col]) and row[col] != 0 ]
	for col in ext_cols_not_null:
		row[col[:-4]] = row[col]
	return row


def determine_tic_onset(row):
	tic_onset_col = 'tic_onset_date'
	default_date = datetime(row['visit_date'].year, 1, 15) # default to visit year (for now) and 15th day (month is arbitrary, should never be unspecified)

	row[tic_onset_col] = None
	if pd.notnull(row['expert_diagnosis_onset']):
		onset_date_str = str(row['expert_diagnosis_onset']).replace('.', '') # remove dots in date string since they cause weird behavior
		row[tic_onset_col] = next(datefinder.find_dates(onset_date_str, base_date=default_date), None)

	# after extractng onset date, make sure that the default year is correct (i.e. that the visit didn't happen at start of new year)
	if row[tic_onset_col] and row[tic_onset_col].year == default_date.year and row[tic_onset_col].month > row['visit_date'].month:
		row[tic_onset_col].replace(year = default_date.year - 1)

	return row[[tic_onset_col]]

def determine_days_since_onset(visit_dates, onset_dates):
	return pd.to_numeric((visit_dates - onset_dates).apply(lambda x: x.days), errors='coerce')


def score_redcap_data(study_name, api_db_password=None, nt_file=None, r01_file=None, use_existing=False):
	with open('cfg/config.json') as config_file:
		config = json.load(config_file)
		study_vars = config[study_name]

	basedir = study_vars['directories']['base'].format(getuser())
	outdir = study_vars['directories']['output'].format(basedir)

	if use_existing:
		return pd.read_csv(join(outdir, RESULT_OUTPUT_FILE.format(study_name))).set_index(NTID)

	db_password = None
	if not nt_file or not r01_file:
		print('No file specified for study - using API instead...')
		db_password = api_db_password

	nt_df = get_dataframe_from_api('nt', config['nt'], db_password) if not nt_file else get_dataframe_from_file(nt_file, config['nt'])
	nt_df['incl_excl_grp'] = 'NT'
	r01_df = get_dataframe_from_api('r01', config['r01'], db_password) if not r01_file else get_dataframe_from_file(r01_file,  config['r01'])
	df = common.merge_projects(nt_df, r01_df)

	if study_name == 'nt':
		df = df[df.index.get_level_values(NTID).isin(nt_df.index.get_level_values(NTID))]

	df = df[~df.index.get_level_values(NTID).isin(EXCLUDED)] # remove excluded/control subjects
	all_events = list(config['nt']['event_name_renames'].values()) + list(config['r01']['event_name_renames'].values())
	df = df[df.index.get_level_values(EVENT_NAME).isin(all_events)] # remove rows (prior to calculation) not used in summaries

	# generate smaller dataframes for each subset of question data
	demo_df = df[get_matching_columns(df.columns, '^demo')].copy()
	demo_df['sex'] = demo_df['demo_sex'].replace([0, 1], ['F', 'M'])
	demo_df = demo_df.apply(score_demographics, axis=1)
	df['visit_date'] = pd.to_datetime(df['visit_date'], errors='coerce')
	demo_df['age'] = (df['visit_date'] - pd.to_datetime(df['demo_dob'])).apply(lambda x: x.days / 365)

	ygtss_df = score_ygtss(df[get_matching_columns(df.columns, '^ygtss.+\d+$')].copy())
	expert_dx_df = score_expert_rated_diagnoses(df[get_matching_columns(df.columns, '^expert_diagnosis')].copy())

	ses_df = score_ses(df[get_matching_columns(df.columns, '^ses')].copy())
	adhd_df = score_adhd(df[get_matching_columns(df.columns, '^adhd.+\d+$')].copy())
	cybocs_df = score_cybocs(df[get_matching_columns(df.columns, '^cybocs.+\d+$')].copy())
	puts_df = score_puts(df[get_matching_columns(df.columns, '^puts.+\d+$')].copy())
	dci_df = score_dci(df[get_matching_columns(df.columns, 'ts_(dci|diagnostic_confidence_index)')].copy(), study_vars) # include both dci column sets
	edinburgh_df = score_edinburgh(df[get_matching_columns(df.columns, '^edinburgh.+\d+$')].copy())
	# need dominant hand from edinburgh frame to map pegboard to handedness
	pegboard_frames = [df[get_matching_columns(df.columns, '^peg.+\d+s$')].copy(), edinburgh_df['dominant_hand']]
	peg_df = score_pegboard(pd.concat(pegboard_frames, axis=1))

	ksads_df = score_ksads(df[get_matching_columns(df.columns, '^ksads')].copy())
	srs_df = score_srs(df[get_matching_columns(df.columns, '^srs')].copy(), demo_df['sex'])
	pedqol_df = score_pedqol(df[get_matching_columns(df.columns, '^pedsql.+\d+$')].copy())

	outcome_df = df[get_matching_columns(df.columns, '^outcome_data_(2|4|5|11)')].copy()
	outcome_df.columns = ['outcome_tics_past_week', 'outcome_doctor', 'outcome_positive_exam', 'outcome_any_3mo_tic_free']
	outcome_df.replace([1,2,3], [False, True, np.nan], inplace=True)

	frames = [ demo_df, ygtss_df, expert_dx_df, srs_df, ses_df, adhd_df, cybocs_df, puts_df, dci_df, ksads_df, edinburgh_df, peg_df, pedqol_df, outcome_df]
	result = pd.concat(frames, axis=1)

	result['kbit_iq'] = df['kbit_iq']
	result['medications'] = df[get_matching_columns(df.columns, '^med_medication_\d')].apply(lambda x: x.str.cat(sep='; ') if pd.notnull(x).any() else np.nan, axis=1)
	result['group'] = df['incl_excl_grp'].replace([1,2,3], ['NT', 'TS', 'HC'])

	# store NTIDs of subjects that have intial screen extra data - need later to fill in screen gaps for these subjects
	has_screen_ext = 'screen_ext' in df.index.get_level_values(EVENT_NAME)
	if has_screen_ext:
		extra_screen_ids = df.xs('screen_ext', level=EVENT_NAME, drop_level=False).index.get_level_values(NTID).values

	result = result.unstack().sort_index(1, level=1) # flatten dataframe such that there is one row per subject per session (all conditions in one line)
	result.columns = [ '_'.join(map(str,i)) for i in result.columns ]
	result = result.dropna(axis=1, how='all')

	# # now that everything is calculated, flattened and all-null columns are removed, trasnfer screen_ext data to screen when available
	# if has_screen_ext:
	# 	# puts check is a hacky solution to handle that there will always be non-null PUTS columns when row is empty
	# 	# 	- safe to do, since PUTS has never happened in a screen_ext visit
	# 	ext_cols = [ col for col in result.columns if col.endswith('_ext') and 'puts' not in col ]
	# 	result.loc[extra_screen_ids] = result.loc[extra_screen_ids].apply(apply_extra_screen_data, args=(ext_cols,), axis=1)
	# 	result = result.drop(ext_cols, axis=1) # remove inital screen columns after data has been copied over

	# get all screen rows of original df (and drop event name from index)
	# 	will be used to calculate things relevant to screen only in flattened df
	screen_only_df = df.xs('screen', level=EVENT_NAME)
	result['1st_relative_with_tics'] = screen_only_df[get_matching_columns(df.columns, '(fh|mafh)\w*_(father|mother|sister|brother)_mvoa.*(1|2)')].sum(axis=1, min_count=1) > 0
	onset_df = screen_only_df[['expert_diagnosis_onset', 'visit_date']].apply(determine_tic_onset, axis=1)
	result.insert(0, 'tic_onset_date', onset_df['tic_onset_date'])
	result.insert(1, 'days_since_onset_screen', determine_days_since_onset(screen_only_df['visit_date'], result['tic_onset_date']))
	result.insert(2, 'days_since_onset_12mo', determine_days_since_onset(df.xs('12mo', level=EVENT_NAME)['visit_date'], result['tic_onset_date']))
	result.insert(3, 'ygtss_total_tic_delta', result['ygtss_past_week_expert_total_tic_12mo'] - result['ygtss_past_week_expert_total_tic_screen'])

	result = multi_session_ever_had(result, 'expert_diagnosis_awareness')
	result = multi_session_ever_had(result, 'dci_attempts_to_suppress_tics')

	twelve_mo_cols = [ col for col in result.columns if col.endswith('_12mo') ]
	result.insert(0, '12mo_complete', result.apply(lambda x: True if not x[twelve_mo_cols].isnull().all() else None, axis=1))

	try:
		result.to_csv(join(outdir, RESULT_OUTPUT_FILE.format(study_name)))
		return result
	except PermissionError:
		raise Exception("Unable to write to output file because it is currently open. Please close it and try again.")


if __name__ == '__main__':
	@Gooey()
	def parse_args():
		parser = GooeyParser(description='Score redcap data (from export file)')
		required = parser.add_argument_group('Required Arguments')
		required.add_argument('--study', required=True, choices=['nt', 'r01'], help='which project to structure data for')

		input = parser.add_argument_group('Data Input Options', gooey_options={'columns':1})
		input.add_argument('--api_db_password', widget='PasswordField')
		input.add_argument('--nt_file', widget='FileChooser', help='file containing data exported from NewTics redcap project (if unspecified API will be used)')
		input.add_argument('--r01_file', widget='FileChooser', help='file containing data exported from R01 redcap project (if unspecified API will be used)')

		return parser.parse_args()

	args = parse_args()
	score_redcap_data(args.study, args.api_db_password, args.nt_file, args.r01_file)
