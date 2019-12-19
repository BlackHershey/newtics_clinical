import os
from cx_Freeze import setup, Executable
import matplotlib

import sys
sys.path.append('..')

PYTHON_PATH = os.path.dirname(sys.executable)
GOOEY_LIB = os.path.join(PYTHON_PATH, 'Lib', 'site-packages', 'gooey')

os.environ['TCL_LIBRARY'] = os.path.join(PYTHON_PATH, 'tcl', 'tcl8.6')
os.environ['TK_LIBRARY'] = os.path.join(PYTHON_PATH, 'tcl', 'tk8.6')

base = "Win32GUI" if sys.platform == "win32" else "console"

executables = [
	Executable('extract_cpt.py', base=base),
	Executable('score_drz.py', base=base),
    Executable('score_weather.py', base=base),
	Executable('score_redcap_data.py', base=base),
	Executable('generate_summary.py', base=base),
    Executable('scan_summary.py', base=base)
]

build_exe_options = {
    'build_exe': 'build',
    'include_files': [
        ('cfg/config.json', 'cfg/config.json'),
        (os.path.join(GOOEY_LIB, 'images'), 'gooey/images'),
        (os.path.join(GOOEY_LIB, 'languages'), 'gooey/languages')
    ],
	'packages': ['common']
}

setup(  name = 'nt_redcap',
        version = "0.1",
        description = 'redcap scripts for NT study',
        options = {'build_exe': build_exe_options},
        executables = executables)
