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
	Executable('redcap_to_nih.py', base=base),
	Executable('merge_image03.py', base=base),
    Executable('check_phi.py', base=base)
]

build_exe_options = {
	'build_exe': 'build',
    'include_files': [
        ('cfg', 'cfg'),
        (os.path.join(GOOEY_LIB, 'images'), 'gooey/images'),
        (os.path.join(GOOEY_LIB, 'languages'), 'gooey/languages')
    ],
	'packages': ['common']
}

setup(  name = 'nih_conversion',
        version = "0.1",
        description = 'scripts to convert nt data to nih submission format',
        options = {'build_exe': build_exe_options},
        executables = executables)
