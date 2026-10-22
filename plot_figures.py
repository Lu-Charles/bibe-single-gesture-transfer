from pathlib import Path
import json, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
B=Path(__file__).resolve().parent;F=B/'figures';P=B
F.mkdir(exist_ok=True)
A=json.loads((P/'revision_analysis.json').read_text());C=json.loads((P/'controls_analysis.json').read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
fig,axes=plt.subplots(2,1,figsize=(3.495,3.65));fig.subplots_adjust(left=.14,right=.98,bottom=.12,top=.84,hspace=.60)
for ax,method,title in zip(axes,['spatial_only','spatial_gain'],['(a) Spatial','(b) Spatial + gain']):
    for j,(key,label,color) in enumerate([('identity_tie','Identity in loss ties','#235A80'),('resample_then_gain','Resample → gain → normalize','#555555')]):
        vals=np.array([p['delta_from_original']*100 for p in C['controls'][key][method]['participants']]);ys=np.arange(4)+(-.17 if j==0 else .17)
        ax.barh(ys,vals,height=.29,label=label,color=color)
        for y,v in zip(ys,vals):
            if abs(v)<.1:ax.plot(v,y,'o',markersize=2,color=color)
    ax.axvline(0,color='black',lw=1.25,zorder=0);ax.set_xlim(-17,6);ax.set_xticks([-15,-10,-5,0,5]);ax.set_yticks(range(4),['P2','P3','P4','P5']);ax.set_ylim(3.5,-.5)
    ax.set_title(title,loc='left',fontsize=9);ax.set_xlabel('Change in accuracy (percentage points)',fontsize=8);ax.tick_params(labelsize=8)
fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',frameon=False,fontsize=8)
for ext in ['png','pdf']:fig.savefig(F/f'participant_controls.{ext}',dpi=600)
plt.close(fig)
G=[8,9,12,13,16,21,23,24]
for method,matrices in A['confusions_by_calibration_gesture'].items():
    fig,axes=plt.subplots(4,2,figsize=(6.7,8.4),layout='constrained')
    for ax,g in zip(axes.flat,G):
        im=ax.imshow(np.array(matrices[str(g)])*100,cmap='Blues',vmin=0,vmax=100)
        ax.set_xticks(range(8),G);ax.set_yticks(range(8),G);ax.tick_params(labelsize=7)
        ax.set_title(f'Calibration gesture {g}',fontsize=9)
        ax.set_xlabel('Predicted provider ID',fontsize=8);ax.set_ylabel('True provider ID',fontsize=8)
        c=G.index(g);ax.add_patch(Rectangle((c-.5,-.5),1,8,fill=False,edgecolor='#D16B2F',linewidth=1.1))
    fig.colorbar(im,ax=axes,shrink=.75,label='Row-normalized trial predictions (%)')
    fig.savefig(F/f'confusion_{method}.png',dpi=220);plt.close(fig)
fig=plt.figure(figsize=(9.8,4.0));ax=fig.add_axes([.02,.07,.96,.88]);ax.axis('off')
boxes=[(.02,.69,.25,.23,'Source session\nAll eight gestures\nTrain source recognizer'),(.36,.69,.27,.23,'Target calibration\nFirst repetition of one gesture\n3 s activity + 90 s rest'),(.73,.69,.25,.23,'Adaptation\nGain / spatial / fine-tuning\nFrozen comparator retained')]
for x,y,w,h,t in boxes:
    ax.add_patch(Rectangle((x,y),w,h,facecolor='#F0F3F5',edgecolor='#444444',lw=.8));ax.text(x+w/2,y+h/2,t,ha='center',va='center',fontsize=10)
for x0,x1 in [(.27,.36),(.63,.73)]:ax.annotate('',(x1,.805),(x0,.805),arrowprops=dict(arrowstyle='->',lw=1.2))
ax.text(.63,.47,'Held-out target trials\nLater repetitions only\nCalibrated gesture and other seven gestures scored separately',ha='center',va='center',fontsize=10)
ax.annotate('',(.83,.56),(.86,.68),arrowprops=dict(arrowstyle='->',lw=1.2))
for row in range(7):
    for col in range(24):ax.add_patch(Rectangle((.02+col*.013,.16+row*.022),.011,.017,facecolor='#235A80',lw=0))
ax.text(.176,.10,'7 × 24 electrode grid',ha='center',fontsize=9)
ax.text(.02,.035,'Provider gesture IDs: 8, 9, 12, 13, 16, 21, 23, 24. Grid shows layout, not electrode displacement.',fontsize=9)
fig.savefig(F/'pipeline.png',dpi=220);plt.close(fig)
print('Figures generated')


fig,ax=plt.subplots(figsize=(3.495,2.65));fig.subplots_adjust(left=.35,right=.985,top=.78,bottom=.20)
methods=['frozen','gain_only','spatial_only','spatial_gain','classifier_fine','classifier_fine_gain']
y=np.arange(6)
for d,key,color,label in [(-.17,'calibrated','#4C4C4C','Calibrated gesture'),(.17,'other','#235A80','Other seven gestures')]:
    ax.barh(y+d,[A['summary'][m][key]*100 for m in methods],height=.28,color=color,label=label)
ax.set_yticks(y,['Frozen','Gain','Spatial','Spatial + gain','Fine-tuning','Fine-tuning + gain']);ax.set_ylim(5.6,-.6);ax.set_xlim(0,102);ax.set_xticks([0,25,50,75,100]);ax.set_xlabel('Held-out trial accuracy (%)')
fig.legend(*ax.get_legend_handles_labels(),loc='upper center',frameon=False)
fig.savefig(F/'calibration_transfer.png',dpi=600);plt.close(fig)
