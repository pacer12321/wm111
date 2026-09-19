"""Incremental video-fidelity check against the frozen parent, not ground truth."""
import argparse,json,math,subprocess
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw
p=argparse.ArgumentParser();p.add_argument('directory',type=Path);args=p.parse_args()
root=args.directory;cv2.setNumThreads(2)
names=['parent','A_whole','B_selective'];caps=[cv2.VideoCapture(str(root/(n+'.mp4'))) for n in names]
info=[dict(frames=int(c.get(cv2.CAP_PROP_FRAME_COUNT)),width=int(c.get(cv2.CAP_PROP_FRAME_WIDTH)),
           height=int(c.get(cv2.CAP_PROP_FRAME_HEIGHT)),fps=c.get(cv2.CAP_PROP_FPS)) for c in caps]
assert all(c.isOpened() for c in caps)
assert all(x==info[0] for x in info),info
N=info[0]['frames'];selected=sorted(set(np.linspace(0,N-1,6,dtype=int).tolist()))
rows=[];metrics={n:[] for n in names[1:]};previous=None

def ssim(a,b):
    # RGB SSIM, 11x11 Gaussian sigma1.5, population variance, valid 5px border.
    a=a.astype(np.float32);b=b.astype(np.float32)
    blur=lambda x:cv2.GaussianBlur(x,(11,11),1.5)
    ma,mb=blur(a),blur(b);va=blur(a*a)-ma*ma;vb=blur(b*b)-mb*mb;co=blur(a*b)-ma*mb
    score=((2*ma*mb+6.5025)*(2*co+58.5225))/((ma*ma+mb*mb+6.5025)*(va+vb+58.5225))
    return float(score[5:-5,5:-5].mean())

for i in range(N):
    got=[c.read() for c in caps];assert all(ok for ok,_ in got),i
    images=[cv2.cvtColor(x,cv2.COLOR_BGR2RGB) for _,x in got]
    floats=[x.astype(np.float32)/255 for x in images]
    for j,name in enumerate(names[1:],1):
        diff=floats[j]-floats[0];mse=float(np.mean(diff*diff))
        item=dict(frame=i,psnr_db=-10*math.log10(max(mse,1e-12)),ssim_rgb=ssim(images[0],images[j]),
                  mean_abs_pixel_difference=float(np.abs(diff).mean()))
        if previous is not None:
            d=(floats[j]-previous[j])-(floats[0]-previous[0])
            item['temporal_difference_rmse_vs_parent']=float(np.sqrt(np.mean(d*d)))
        metrics[name].append(item)
    previous=floats
    if i in selected:
        strip=Image.new('RGB',(1344,280),'#202020');draw=ImageDraw.Draw(strip)
        for j,name in enumerate(names):
            strip.paste(Image.fromarray(images[j]).resize((448,256)),(j*448,24))
            draw.text((j*448+8,5),f'{name} / frame {i}',fill='white')
        rows.append(strip)
        if i in (selected[2],selected[-2]):
            panel=Image.new('RGB',(info[0]['width']*3,info[0]['height']+28),'#202020')
            d=ImageDraw.Draw(panel)
            for j,name in enumerate(names):
                panel.paste(Image.fromarray(images[j]),(j*info[0]['width'],28))
                d.text((j*info[0]['width']+8,6),name,fill='white')
            panel.save(root/f'full_frame_{i:03d}.jpg',quality=95)
    if i%25==0:print(json.dumps(dict(stage='frames',completed=i+1,total=N)),flush=True)
sheet=Image.new('RGB',(1344,280*len(rows)))
for i,row in enumerate(rows):sheet.paste(row,(0,i*280))
sheet.save(root/'contact_sheet.jpg',quality=94)
summary={}
for name,values in metrics.items():
    summary[name]={}
    for key in ('psnr_db','ssim_rgb','mean_abs_pixel_difference','temporal_difference_rmse_vs_parent'):
        v=np.array([r[key] for r in values if key in r])
        summary[name][key]=dict(mean=float(v.mean()),min=float(v.min()),max=float(v.max()),p10=float(np.quantile(v,.1)))
report=dict(reference='209.870141s step1000 KVreuse + T skip; NOT ground-truth target',
    note='Same seed/weights/input but each run owns its bootstrap selector; single-run incremental fidelity, not formal quality acceptance.',
    video=info[0],summary=summary,per_frame=metrics,
    temporal_metric='Unwarped difference-of-frame-differences relative to parent; not a motion-compensated flicker metric')
(root/'quality.json').write_text(json.dumps(report,indent=2))
print(json.dumps(dict(stage='completed',summary=summary)),flush=True)
filters=[]
for j,name in enumerate(names):
    filters.append(f'[{j}:v]scale=672:384,setsar=1,drawtext=text={name}:x=10:y=10:fontsize=20:fontcolor=white:box=1:boxcolor=black@0.65[v{j}]')
filters.append('[v0][v1][v2]hstack=inputs=3[out]')
cmd=['ffmpeg','-n','-v','error','-threads','2']
for name in names:cmd+=['-i',str(root/(name+'.mp4'))]
cmd+=['-filter_complex_threads','2','-filter_complex',';'.join(filters),'-map','[out]',
      '-an','-c:v','libx264','-threads','2','-crf','18',str(root/'comparison.mp4')]
subprocess.run(cmd,check=True)
