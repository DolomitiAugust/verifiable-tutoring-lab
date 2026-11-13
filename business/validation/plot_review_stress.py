"""Reproducible business review-cost sensitivity; deterministic assumptions only."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
data=json.loads((HERE/'results/quality_cost_stress.json').read_text(encoding='utf-8'))
rows=data['rows']; plt.rcParams.update({'font.family':['Times New Roman','SimSun'],
    'font.size':9,'axes.spines.top':False,'axes.spines.right':False,
    'pdf.fonttype':42,'axes.unicode_minus':False})
fig,(left,right)=plt.subplots(1,2,figsize=(6.2,3.1))
x=[r['review_fraction']*100 for r in rows]
y=[r['contribution_after_incremental_cac_cny']/10000 for r in rows]
zero=-y[0]*100/(y[-1]-y[0])
for ax in (left,right):
    ax.plot(x,y,color='#0072B2',marker='o',markersize=3.5,linewidth=1.5)
    ax.axhline(0,color='#555555',linewidth=.8,linestyle='--')
    ax.grid(alpha=.15)
left.set(xlim=(-2,102),ylim=(-30,7),xticks=[0,25,50,75,100],ylabel='获客后贡献（万元 / 全年新合同）')
left.set_title('A  完整压力范围',loc='left',fontweight='bold')
right.set(xlim=(-.7,25.7),ylim=(-3.5,7),xticks=[0,5,10,15,20,25],yticks=[-2,0,2,4,6])
right.set_title('B  关键区间放大',loc='left',fontweight='bold')
right.axvline(zero,color='#D55E00',linestyle=':',linewidth=1.2)
right.scatter([zero],[0],color='#D55E00',marker='D',s=20,zorder=5)
right.annotate(f'{zero:.2f}%：贡献为零',xy=(zero,0),xytext=(5,-2.4),
               arrowprops={'arrowstyle':'-','lw':.7,'color':'#555555'},fontsize=8)
right.annotate('基准 2%',xy=(2,y[1]),xytext=(5,5.5),arrowprops={'arrowstyle':'-','lw':.7},fontsize=8)
fig.text(.50,.12,'需人工复核的轮次比例（假设，%）',ha='center',fontsize=9)
fig.text(.12,.025,'仅为原定价及工时假设下的计算；模型回退率不等于人工复核率。',fontsize=7.5)
fig.subplots_adjust(left=.13,right=.98,top=.84,bottom=.28,wspace=.35)
dest=HERE.parents[2]/'report'/'figures'/'business-review-stress'
fig.savefig(str(dest)+'.pdf');fig.savefig(str(dest)+'.png',dpi=300);plt.close(fig)
print(dest)
