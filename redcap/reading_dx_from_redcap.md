# Converting REDCap tic diagnosis data to R01_combine and ENIGMA-TS files

## Files of interest
- Box\Black_Lab\projects\TS\New_Tics_R01\Data\analysis\DS\r01_combine\reading_dx_from_redcap.md is this file
- Box\Black_Lab\projects\TS\ENIGMA-TS\phenotyping\ENIGMA-TS_spreadsheet_template_2024-08-01.xlsx contains the desired format, with the first tab ("DICTIONARY") giving the rules, data formats, etc.

## Fields used in ENIGMA-TS spreadsheet

As the DICTIONARY tab indicates, all the diagnoses go in the scan_day sheet. 
The *current* diagnosis fields are the ones we're concerned with here as coming directly from REDCap.

| Spreadsheet |           Fields           | Optional | Explanation |
| ------------|----------------------------|----------|-------------|
| scan_day    | tic_diagnosis_current      | | |
| scan_day    | tic_dx_current_criteria    | | |
| scan_day    | tic_dx_current_source      | | |
| scan_day    | tic_dx_current_date        | | |


## Study visits in the (pre-R01) NewTics database

Information is entered in the "Expert Rated Diagnoses" form unless specifically mentioned below.

Field names used: 
- expert_diagnosis_tourette 		present = 1, subthreshold = 2, not present = 3, blank = 4
- expert_diagnosis_chronic_tics 	same codes  n.b. label says "chronic *motor* tics DSM-5"
- expert_diagnosis_transient		same codes



## Study visits in the New Tics R01 database

Information is entered in the "Expert Rated Diagnoses" form unless specifically mentioned below.

Field names used: 
- expert_diagnosis_tourette 		present = 1, subthreshold = 2, not present = 3, blank = 4
- expert_diagnosis_chronic_tics 	same codes  n.b. label says "chronic *motor* tics DSM-5"
- expert_diagnosis_transient		same codes
- incl_excl_grp (from the Inclusionexclusion Checklist form) 1=NT, 2=TS/CTD, 3=TFC

**RULES**

**Ensure all fields below come from the correct visit, i.e. screen or 12mo.**
```python
if expert_diagnosis_tourette == 1 :
	tic_diagnosis_current = "TS" 
elif expert_diagnosis_chronic_tics == 1 :
	tic_diagnosis_current = "CMTD" 
elif expert_diagnosis_transient = 1:
	tic_diagnosis_current = "PTD" 
elif expert_diagnosis_tourette == 2:   # subthreshold
	tic_diagnosis_current = "TS" 
elif expert_diagnosis_chronic_tics == 2:
	tic_diagnosis_current = "CMTD" 
elif expert_diagnosis_transient == 2:
	tic_diagnosis_current = "PTD"
# at this point, none of the tic diagnoses are selected, but we need to check
elif {TODO: code here that means this study visit is in the pre-R01 database} and ((incl_excl_control == 1) or 
		(incl_excl_control == 2)):  # i.e. answered this question that was for control subjects only
	# incl_excl_control is from the "Inclusionexclusion Checklist" form, in the pre-R01 database only
	tic_diagnosis_current = "none"   # i.e. enrolled as control, no tic diagnosis selected above
elif {TODO: code here that means this study visit is in the New Tics R01 database} and incl_excl_grp == 3:   
	# incl_excl_grp is from the "Inclusionexclusion Checklist" form, in the New Tics R01 database only
	tic_diagnosis_current = "none"   # i.e. enrolled as control, no tic diagnosis selected above
else  # could be that there are comments that explain what's up
	print(f'\n{this_subject}-{this visit} needs human review')  
	# TODO: replace names in braces in the line above with real variable names
	if incl_excl_grp == 1:
		print('NT group at screening')
	elif incl_excl_grp == 2:  
		print('TS/CTD group at screening')
	elif incl_excl_grp == 3:
		print('TFC group at screening')
	else:
		print('incl/excl form doesn't specify dx group at screening')
	print('Comments 1:', expert_diagnosis_comments)   # e.g. could be CVTD or Other Tic Disorder
	print('Comments 2:', expert_diag_comments)
# now fill in the other fields for current diagnosis:
tic_dx_current_criteria = 'DSM-5'
tic_dx_current_source = clinician
tic_dx_current_date = visit_date  # from 'Visit Info" form **at this visit**
```
