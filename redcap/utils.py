import json
import os.path

from getpass import getuser


def get_study_vars(study_name):
	with open('cfg/config.json') as config_file:
		config = json.load(config_file)

	return config, config[study_name]


def get_scoring_dirs(study_vars, assessment):
	basedir = study_vars['directories']['base'].format(getuser())
	indir = study_vars['directories']['input'][assessment].format(basedir)
	outfile = os.path.join(indir, '{}_result.csv'.format(assessment.lower()))

	return basedir, indir, outfile


def get_screen_event_col(study_vars):
	for k,v in study_vars['event_name_renames'].items():
		if v == 'screen':
			return k + '_arm_1'
	return None
