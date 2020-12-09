import subprocess as sp
import os
from joblib import Parallel, delayed
import tqdm

import sys
if len(sys.argv) == 2:
  every_k = sys.argv[1]
else:
  every_k = 10

expname = 'lego_coarse_nerflet_test'
def write_im(idx, i):
  print(f'Processing {idx}, {i}')
  outdir = f'/home/kgenova/nerflet/logs/{expname}/qvid'
  if not os.path.isdir(outdir):
    os.mkdir(outdir)
  out_path = f'{outdir}/{str(i).zfill(6)}.png'
  cmd = f'qview /home/kgenova/nerflet/logs/{expname}/sif_{str(idx).zfill(6)}.txt -camera 2.8018 3.33692 4.7267  -0.443111 -0.459386 -0.769816  -0.209809 0.888016 -0.409154 -show_axes -image {out_path}'
  sp.check_output(cmd, shell=True)

idx = 10
i = 0

idxs = []
inds = []
while True:
  print(f'Idx: {idx}')
  path = f'/home/kgenova/nerflet/logs/{expname}/sif_{str(idx).zfill(6)}.txt'
  if not os.path.isfile(path):
    print(f'No sif {path}')
    break
  #write_im(idx, i)
  idxs.append(idx)
  inds.append(i)
  idx += every_k
  i += 1
to_proc = list(zip(idxs, inds))
print(to_proc)
for p in tqdm.tqdm(to_proc):
  write_im(*p)
#Parallel(n_jobs=1, backend='threading')(delayed(write_im(*a))(a) for a in tqdm.tqdm(to_proc))

outvid = f'/home/kgenova/nerflet/logs/{expname}/sifs.mp4'
cmd = f'ffmpeg -y -i /home/kgenova/nerflet/logs/{expname}/qvid/%06d.png {outvid}'
#cmd = f'convert -delay 10 -loop 1 /home/kgenova/nerflet/logs/{expname}/qvid/*.png /home/kgenova/nerflet/logs/{expname}/sifs.gif'
sp.check_output(cmd, shell=True)
sp.check_output(f'mplayer {outvid} -loop 0', shell=True)

