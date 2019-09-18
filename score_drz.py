import csv
import itertools
import numpy as np
import re
import pandas as pd

from datetime import datetime, time, timedelta
from glob import glob
from gooey import Gooey, GooeyParser
from os import getcwd, listdir
from os.path import join, basename

RESULT_OUTPUT_FILE = 'drz_output.csv'
CHECK_OUTPUT_FILE_PREFIX = 'drz_missing_and_extra_data'
INDEX_COLS = ['demo_study_id', 'event_name']

DEFAULT_FILENAME_FORMAT = 'NT\d{3}_(screen|\d{2}mo)_session\d{1}\w{0,1}_(baseline|verbal|DRZ|NCR)_TicTimer_log.txt'

def extract_time(line):
	time_str = re.search('\d{2}:\d{2}:\d{2}', line).group()
	return datetime.strptime(time_str, '%H:%M:%S')


def get_drz_files(directory, filename_format, nested=True):
	files = [ f for f in glob('{}/**'.format(directory), recursive=True) ] if nested else [ join(directory, f) for f in listdir(directory) ]
	return [ f for f in files if re.search(filename_format, f, flags=re.IGNORECASE) ]


def parse_drz(indir, filename_format, nested, use_existing, check_missing):
	if use_existing:
		return pd.read_csv(join(indir, RESULT_OUTPUT_FILE)).set_index(INDEX_COLS)

	drz_files = get_drz_files(indir, filename_format, nested)
	print(len(drz_files))
	results = []
	for filename in drz_files:
		print('Processing', filename)

		subject, event_name, session, condition = basename(filename).split('_')[0:4]
		subject = subject.upper() # capatalize all for correct grouping when scoring

		f = open(filename, 'r')

		start_times = []
		end_times = []
		tic_time = [0]
		total_tics = [0]
		total_rewards = [0]
		previous_line = None
		duration = None
		for line in f:
			if 'Session over' in line:
				session_end = extract_time(previous_line)
				duration = time(0, 5, 0) if (session_end.time() >= time(0,4,50) and session_end.time() <= time(0, 5, 0)) else session_end.time()
				notes = 'Ends with session over'
			if 'began at' in line:
				if len(end_times) < len(start_times):
					end_times.append(None)
				start_times.append(extract_time(line))
				if len(start_times) > 1:
					total_tics.append(0)
					total_rewards.append(0)
					tic_time.append(0)
			if  re.search('Session \w+ ended at', line):
				if len(start_times) < len(end_times):
					start_times.append(None)
				end_times.append(extract_time(line))
			if 'Tic detected' in line:
				total_tics[len(total_tics)-1] += 1
				if 'Lite' in filename:
					time2 = int(re.search('(\d+) ms', line).group(1))
					total_rewards[len(total_rewards)-1] += (time2 - tic_time[len(total_rewards)-1]) // 10000
					tic_time[len(total_rewards)-1] = time2

			elif re.search('(Reward sent|10s tic free|No tics)', line): # handles both old and new format of TicTimer logs
				total_rewards[len(total_rewards)-1] += 1
			previous_line = line

		f.close()

		notes = ''
		if duration:
			total_tics = total_tics[-1]
			total_rewards = total_rewards[-1]
			duration = timedelta(minutes=duration.minute, seconds=duration.second)
		else:
			drop_idx = [ i for i,v in enumerate(zip(start_times, end_times)) if None in v ]
			if drop_idx or len(start_times) != len(end_times):
				notes = 'Length mismatch for start and end times: {}, {}'.format(start_times, end_times)

			if len(drop_idx) < len(start_times):
				runtimes = [ (end-start) for start, end in [ tup for i, tup in enumerate(zip(start_times, end_times)) if i not in drop_idx ] ]
				total_tics = [ val for i, val in enumerate(total_tics) if i not in drop_idx ]
				total_rewards = [ val for i, val in enumerate(total_rewards) if i not in drop_idx ]
				tic_time = [ val for i, val in enumerate(tic_time) if i not in drop_idx ]
			else:
				runtimes = [ timedelta(minutes=5) ]

			if len(runtimes) > 1:
				if runtimes[-1] >= timedelta(minutes=4, seconds=50):
					runtimes = runtimes[-1:]
					total_tics = total_tics[-1:]
					total_rewards = total_rewards[-1:]
					tic_time = tic_time[-1:]
					notes = 'Multiple start/stop times found; used final run (it was expected length)'
				else:
					notes = 'Multiple start/stop times found; none were of expected length; used all runs in file'

			if 'Lite' in filename and total_rewards:
				for i, t in enumerate(runtimes):
					total_rewards[i] += (runtimes[i].seconds - (tic_time[i] / 1000)) // 10

			duration = duration if duration else np.sum(runtimes)
			total_tics = np.sum(total_tics) if type(total_tics) is list else total_tics
			total_rewards = np.sum(total_rewards) if type(total_rewards) is list else total_rewards

		if total_tics == 0 and total_rewards == 0: # if file didn't contain data, exclude it from df (don't want 0 averaged in)
			if 'Lite' in filename:
				total_rewards = duration.seconds // 10 if duration else None
			else:
				print('Found file for {} {} {} but it didn\'t seem to contain any data'.format(subject, event_name, condition))
				continue
		results.append([subject, event_name, session[7:], condition, total_tics, total_rewards, (duration.total_seconds() / 60) if duration else np.nan, filename, notes])

	columns = ['demo_study_id', 'event_name', 'session', 'condition', 'tics', 'rewards', 'duration', 'file', 'notes']
	df = pd.DataFrame(results, columns=columns).set_index(INDEX_COLS)

	if check_missing:
		missing_data = df.groupby(['demo_study_id', 'event_name', 'condition']).apply(lambda x: len(x) if len(x) != 2 else None).dropna()
		missing_data.to_csv(CHECK_OUTPUT_FILE_PREFIX + '_' + datetime.now().strftime('%m-%d_%H%M') + '.csv') # save to file with current datetime in name

	df.to_csv(join(indir, RESULT_OUTPUT_FILE))
	return df.drop(columns=['file'])


def score_drz(indir, filename_format=DEFAULT_FILENAME_FORMAT, nested=False, use_existing=False, check_missing=False):
	df = parse_drz(indir, filename_format, nested, use_existing, check_missing)

	avg_tpm = 'avg_tics_per_minute'
	avg_tfipm = 'avg_tic_free_10s_per_minute'

	df = df.groupby(['demo_study_id', 'event_name', 'condition']).mean() # calculate average tics/rewards for each condition
	df = df.rename(columns={'tics': avg_tpm, 'rewards': avg_tfipm})
	df = df[[avg_tpm, avg_tfipm]].div(df['duration'], axis=0)

	df = df.unstack() # flatten dataframe such that there is one row per subject per session (all conditions in one line)
	df.columns = [ '_'.join(map(str,i)) for i in df.columns ]

	print(df)
	# generate list of ordered column names by combining each measure with each condition
	measures = [avg_tpm, avg_tfipm]
	conditions = ['baseline', 'verbal', 'DRZ', 'NCR'] # in order of first 4 sessions
	ordered_columns = [ '_'.join([a,b]) for a, b in list(itertools.product(measures, conditions)) ]
	df = df[[ col for col in ordered_columns if col in df ]] # reorder columns to fit expected output (all avg_tpm, followed by all avg_tfipm); NCR may or may not be present

	df = df.unstack().sort_index(1, level=1) # flatten so that there is on participant per row
	df.columns = [ '_'.join(map(str,i)) for i in df.columns ]

	df[avg_tpm + '_verbal_delta_screen'] = df[avg_tpm + '_verbal_screen'] - df[avg_tpm + '_baseline_screen']
	df[avg_tfipm + '_verbal_delta_screen'] = df[avg_tfipm + '_verbal_screen'] - df[avg_tfipm + '_baseline_screen']

	return df


if __name__ == '__main__':
	# set up expected arguments and associated help text
	@Gooey()
	def parse_args():
		parser = GooeyParser(description='parses and scores drz files')

		required = parser.add_argument_group('Required Arguments', gooey_options={'columns':1})
		required.add_argument('--indir', widget='DirChooser', required=True, help='directory containing drz txt files')

		optional = parser.add_argument_group('Optional Arguments', gooey_options={'columns':1})
		optional.add_argument('--filename_format', default=DEFAULT_FILENAME_FORMAT, help='regex for matching filenames in indir')
		optional.add_argument('-r', '--reuse', action='store_true', help='reuse the output file from previous run (default is to parse again)')
		optional.add_argument('--check', action='store_true', help='check for missing/extra data and output anomalies to file (default is not to check)')
		optional.add_argument('--nested', action='store_true', help='if indir has log files nested in subject directories')
		return parser.parse_args()


	args = parse_args()
	print(args)

	score_drz(args.indir, args.filename_format, args.nested, args.reuse, args.check)
