import os, shutil
src_dir = 'D:/wmed/emsim/pdf/Endocrine/'
for f in os.listdir(src_dir):
    if 'CAH' in f:
        print('Found:', repr(f))
        # Don't rename source - just copy to an ASCII name
        dst = os.path.join(src_dir, 'CAH_TEMP_ASCII.pdf')
        shutil.copy(os.path.join(src_dir, f), dst)
        print('Copied to:', dst)
