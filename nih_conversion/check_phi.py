import numpy as np
import os
import pandas as pd
import pyodbc
import re
import sys

from datetime import datetime
from gooey import Gooey, GooeyParser

sys.path.append('..')
from common import get_redcap_project

DB_PATH = r'\\neuroimage\nil\blackf\K\Projects\TS\NewTics\Recruitment\NewTics_recruit_forms.accdb'

"""
Query recruitment database for subject/parent names
"""
def get_phi_lut(password):
	try:
		conn_str = (
			r'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};'
			r'DBQ=' + DB_PATH + ';'
			r'PWD=' + password
		)
		conn = pyodbc.connect(conn_str)
	except pyodbc.Error:
		sys.stderr.write('Error connecting to Recruitment access database')
		sys.exit(1)

	sql = 'SELECT Contacts.FirstName, Contacts.LastName, Contacts.Guardian1name, Contacts.Guardian2name' + \
			' FROM Contacts' + \
			' WHERE Contacts.NewTicsID IS NOT NULL'
	return pd.read_sql(sql, conn)


def check_phi(df, lookup_string):
	for col in df.columns:
		# for each column, search for a name surrounded by word boundaries or followed by possessive 
		#	extra checks are to ensure names that are common word prefixes or contractions don't get picked up (otherwise lots of extra fields to check)
		df[col] = df[col].apply(lambda x: x if pd.notnull(x) and re.search(r'\b({})(\s|\'s)'.format(lookup_string), str(x), flags=re.IGNORECASE) else np.nan)
	df = df.dropna(how='all', axis=1).dropna(how='all', axis=0) # drop row and columns that are completely empty (again, limits what needs to be visually checked)
	return df

"""
Generate file with cells that possibly contain subject PHI -- issueswill need to be corrected in REDCap prior to data sharing
"""
def flag_redcap_phi(study_key, outdir, password, from_date):
	# get redcap dataframe (from API) and filter out rows we've checked previously (via optional date param)
	project = get_redcap_project(study_key, password)
	df = project.export_records(format='df').dropna(how='all', axis=0).drop(columns='demo_dob')
	df['visit_date'] = pd.to_datetime(df['visit_date'])
	df = df[df['visit_date'] > from_date]

	# get all possible names to check 
	lut = get_phi_lut(password)
	lut = lut.fillna('')
	lookup_string = '|'.join({ x for x in filter(str.isalpha, lut.to_string().split()) }) # create regex compatible match string with names

	print('beginning checks...')
	df = check_phi(df, lookup_string)

	df.to_csv(os.path.join(outdir, '{}_phi_issues.csv'.format(study_key))) # save to file


if __name__ == '__main__':
	@Gooey
	def parse_args():
		parser = GooeyParser()
		parser.add_argument('study', choices=['nt','r01'])
		parser.add_argument('outdir', widget='DirChooser', help='where to store output CSV')
		parser.add_argument('db_password', widget='PasswordField', help='password for recruitment database')
		parser.add_argument('--from_date', widget='DateChooser', type=lambda d: datetime.strptime(d, '%Y-%m-%d'), help='only check visits from this date on')
		return parser.parse_args()

	args = parse_args()
	flag_redcap_phi(args.study, args.outdir, args.db_password, args.from_date)
