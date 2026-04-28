import os, shutil, sys
d = 'D:/wmed/emsim/pdf/Endocrine/'
for f in os.listdir(d):
    if 'CAH' in f:
        src = d + f
        dst = d + 'CAH_ASCII.pdf'
        if not os.path.exists(dst):
            shutil.copy(src, dst)
        print('Original:', repr(f))
        print('Copy to:', dst)
        print('Exists:', os.path.exists(dst))
