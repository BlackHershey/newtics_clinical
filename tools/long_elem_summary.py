import argparse
import itertools
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re

from sklearn.metrics import cohen_kappa_score

tic_types = [ 'motor', 'vocal', 'simple', 'complex' ]
study_dir = r'C:\Users\acevedoh\Box\Black_Lab\projects\TS\NewTics\Data\Raw_Data\classroom_observations\analysis'

def is_persistent(group):
	num_months = group.dropna().sum()
	return num_months == 3 or (num_months == 2 and 4 not in group.index.levels[1])


def get_summary_df(tic_data):
	temp = []
	for mnth, mnth_data in tic_data.groupby('month'):
		for stud, stud_data in mnth_data.groupby('student'):
			present = stud_data['present'].any()
			location_yn = []
			for loc in tic_locations:
				location_yn.append(any(loc in x for x in stud_data['location'] if pd.notnull(x)))
			motor_yn = any('vocal' not in x for x in stud_data['location'] if pd.notnull(x))
			vocal_yn = any('vocal' in x for x in stud_data['location'] if pd.notnull(x))
			simple_yn = any('S' in x for x in stud_data['simple_or_complex'] if pd.notnull(x))
			complex_yn = any('C' in x for x in stud_data['simple_or_complex'] if pd.notnull(x))

			temp.append([stud, int(mnth), stud_data.iloc[0]['grade'], present, motor_yn, vocal_yn, simple_yn, complex_yn] + location_yn)

	index_cols = ['student', 'month']
	return pd.DataFrame(temp, columns=index_cols + ['grade', 'present'] + tic_types + tic_locations).set_index(index_cols)


def get_point_prev(df):
	monthly_pp = [['tic_type', 'month', 'pt_prev']]
	for type in tic_types:
		# print( df.groupby('month').apply(lambda g: [g[type].sum(), g['present'].sum()]))
		pt_prev = df.groupby('month').apply(lambda g: 100* (g[type].sum() / g['present'].sum())).tolist()
		monthly_pp += [ [type, i, pp] for i, pp in enumerate(pt_prev) ]
	pp_df = pd.DataFrame(monthly_pp[1:], columns=monthly_pp[0]).set_index(['tic_type', 'month']).unstack(level=0)
	pp_df = pp_df[[('pt_prev', type) for type in tic_types ]]

	return pp_df


def get_tic_characteristics(df):
	tic_char = [['tic_type', 'group', 'N', 'no_tics', 'isolated', 'persistent']]
	for type in tic_types:
		for grpby in ['grade']: #[ 'gender', 'grade' ]:
			for grp, data in df.groupby(grpby):
				grp_n = data.groupby('student')['present'].any().sum()
				persistent_n = data.groupby('student')[type].apply(is_persistent).sum()
				# no_tics_n = data.groupby('student').apply(lambda g: pd.notnull(g['present']).any() and g[type].sum() == 0).sum()
				# That row counted a kid with 3 n/a equivalents for 3 visits, and no "location" or "simple_or_complex" entries.
				no_tics_n = data.groupby('student').apply(lambda g: (g['present'].any() and g[type].sum() == 0).sum()
				isolated_n = grp_n - no_tics_n - persistent_n
				tic_char.append([type, grp, grp_n, no_tics_n, isolated_n, persistent_n])

	return pd.DataFrame(tic_char[1:], columns=tic_char[0])


def write_to_sheet(df, writer, sheet_name, header=True):
	df.to_excel(writer, sheet_name=sheet_name, header=header)

parser = argparse.ArgumentParser()
parser.add_argument('--observer', type=int)
args = parser.parse_args()

trailer = '_observer{}'.format(args.observer) if args.observer else ''

tic_data = pd.read_excel(os.path.join(study_dir, 'classroom_observations_20190509_HA_fmt.xlsx'))
tic_data = tic_data.drop(columns=['classroom', 'notes', 'tic_list'])
tic_data = tic_data.set_index(['student', 'grade', 'date_time', 'tics_observed_yn', 'confidence', 'observer'])
tic_data.columns = tic_data.columns.str.rsplit('_', 1, expand=True)
tic_data = tic_data.stack(dropna=False).reset_index()

tic_data['month'] = tic_data['date_time'].str[0]
tic_data['present'] = tic_data['tics_observed_yn'].apply(lambda x: pd.notnull(x) and ('Y' in x or 'N' in x))
tic_data['location'] = tic_data['location'].replace(['nose/mouth', 'brows'], ['mouth/nose', 'eyebrow'])
tic_data = tic_data.dropna(subset=['observer'])
if args.observer:
	tic_data = tic_data[tic_data['observer'] == args.observer]

N = tic_data.groupby('student')['present'].any().sum()
tic_locations = [ loc for loc in tic_data['location'].unique() if loc != 'vocal' and pd.notnull(loc) ]

all_tics_df = get_summary_df(tic_data)
def_tics_df = get_summary_df(tic_data[tic_data['could_be_other'] == 'no'])
missing_rows = all_tics_df[~all_tics_df.index.isin(def_tics_df.index)][['grade','present']]
def_tics_df = pd.concat([def_tics_df, missing_rows], axis=0).fillna(False)

### get + plot point prevalence
all_pp_df = get_point_prev(all_tics_df)
def_pp_df = get_point_prev(def_tics_df)

fig, axes = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True)

all_pp_df.plot.bar(ax=axes[0], legend=None)
axes[0].set_title('All observed tics')
axes[0].set_ylabel('Point prevalence of tics (%)')

def_pp_df.plot.bar(ax=axes[1])
axes[1].set_title('Definite tics')

plt.xlabel('Month observed')
plt.ylabel('Point prevalence of tics (%)')
plt.xticks(ticks=range(3), labels=['March', 'April', 'May'])
plt.legend(tic_types)
plt.savefig(os.path.join(study_dir, 'monthly_point_prevalence{}.png'.format(trailer)))


# tic characteristics (isolated/persistent)
all_char_df = get_tic_characteristics(all_tics_df)
def_char_df = get_tic_characteristics(def_tics_df)

# tics by body part ever
body_part_df = all_tics_df.groupby('student')[tic_locations].any().sum()

# calculate other frequencies of interest
other = {}
other['Missed a month'] = (~tic_data.groupby('student')['present'].all()).sum()

obs_numtics = tic_data.groupby(['student','month','observer'])['location'].count()
obs_2plus = obs_numtics > 1
month_2plus = obs_2plus.groupby(['student', 'month']).any()
other['2+ tics (any visit)'] = month_2plus.groupby('student').any().sum()
other_df = pd.DataFrame.from_dict(data=other, orient='index')

# calculate inter-observer reliability
obs_hastics = obs_numtics > 0
obs_hastics = obs_hastics.unstack()
irr = [['observer_pair', 'N', 'kappa']]
for pair in itertools.combinations(obs_hastics.columns, 2):
	temp = obs_hastics[list(pair)].dropna(how='any').astype(int)
	if temp.empty:
		continue
	irr.append([pair, len(temp), cohen_kappa_score(temp[pair[0]], temp[pair[1]])])
irr_df = pd.DataFrame(irr)


### write whole summary to file
with pd.ExcelWriter(os.path.join(study_dir, 'summary{}.xlsx'.format(trailer)), engine='xlsxwriter') as writer:
	write_to_sheet(all_pp_df, writer, sheet_name='Point Prevalence (all)')
	write_to_sheet(def_pp_df, writer, sheet_name='Point Prevalence (definite)')
	write_to_sheet(all_char_df, writer, sheet_name='Persistent (all)')
	write_to_sheet(def_char_df, writer, sheet_name='Persistent (definite)')
	write_to_sheet(body_part_df, writer, sheet_name='Body Part ever', header=False)
	write_to_sheet(other_df, writer, sheet_name='Other stats', header=False)
	write_to_sheet(irr_df, writer, sheet_name='IRR (kappa)', header=False)
