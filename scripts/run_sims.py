import glob
import os

files = glob.glob('just_linear_algebra_things_*_*.py')

for infile in files:
    print(f'Running {infile}...')
    os.system(f'python {infile}')
