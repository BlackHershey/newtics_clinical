# Modified version of Kevin's jypter notebook Descriptor
# Changed to be python 3 compatible

import xlrd # http://www.python-excel.org/
import csv

# from http://stackoverflow.com/questions/26029095/python-convert-excel-to-csv
def excel_to_csv(ExcelFile, CSVFile, SheetIndex=0, SheetName=None):
    workbook = xlrd.open_workbook(ExcelFile)
    if SheetName is not None:
        worksheet = workbook.sheet_by_name(SheetName)
    else:
        worksheet = workbook.sheet_by_index(SheetIndex)
    csvfile = open(CSVFile, 'w', newline='\n') # open for writing (truncating the file if it already exists)
    wr = csv.writer(csvfile) # (default quoting=csv.QUOTE_MINIMAL) , quoting=csv.QUOTE_NONNUMERIC
    for rownum in range(worksheet.nrows):
        wr.writerow(worksheet.row_values(rownum))
    csvfile.close()
