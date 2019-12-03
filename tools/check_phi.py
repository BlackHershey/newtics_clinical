import argparse
import numpy as np
import os
import pandas as pd
import pyodbc
import re
import sys

from datetime import datetime
from getpass import getpass

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'redcap'))
from score_redcap_data import get_redcap_project

DB_PATH = r'\\neuroimage\nil\blackf\K\Projects\TS\NewTics\Recruitment\NewTics_recruit_forms.accdb'

def get_phi_lut(password):
	try:
		conn_str = (
			r'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};'
			r'DBQ=' + DB_PATH + ';'
			r'PWD=' + password
		)
		conn = pyodbc.connect(conn_str)
	except pyodbc.Error:
		sys.stderr.write('Error connecting to access database')
		sys.exit(1)

	sql = 'SELECT Contacts.FirstName, Contacts.LastName, Contacts.Guardian1name, Contacts.Guardian2name, Contacts.Birthdate' + \
			' FROM Contacts' + \
			' WHERE Contacts.NewTicsID IS NOT NULL';
	return pd.read_sql(sql, conn)


def flag_redcap_phi(study_key, from_date):
	password = getpass('db password:')
	project = get_redcap_project(study_key, password)
	df = project.export_records(format='df').dropna(how='all', axis=0).drop(columns='demo_dob')
	df['visit_date'] = pd.to_datetime(df['visit_date'])
	df = df[df['visit_date'] > from_date]

	lut = get_phi_lut(password)
	lut = lut.drop(columns='Birthdate').fillna('')
	lookup_string = '|'.join({ x for x in filter(str.isalpha, lut.to_string().split()) })

	print('beginning checks...')
	for col in df.columns:
		df[col] = df[col].apply(lambda x: x if pd.notnull(x) and re.search(r'\b({})(\s|\'s)'.format(lookup_string), str(x), flags=re.IGNORECASE) else np.nan)
	df.dropna(how='all', axis=1).dropna(how='all', axis=0).to_csv('{}_phi_issues.csv'.format(study_key))


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('study', choices=['nt','r01'])
	parser.add_argument('--from_date', type=lambda d: datetime.strptime(d, '%Y-%m-%d'), help='only check visits from this date on')

	args = parser.parse_args()

	flag_redcap_phi(args.study, args.from_date)
