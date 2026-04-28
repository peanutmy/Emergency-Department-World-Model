import os, sys, glob
matches = glob.glob('D:/wmed/emsim/pdf/Endocrine/CAH*')
for m in matches:
    print('FOUND:', repr(m))
    print('SIZE:', os.path.getsize(m))
