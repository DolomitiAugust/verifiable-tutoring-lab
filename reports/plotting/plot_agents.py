"""Plot role adoption and evidence-only memory from the completed real run."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'reports/assets/zh'
result = json.loads((ROOT / 'tutor/legacy/results/core/results.json').read_text(encoding='utf8'))
assert result['completed']
turns = [t['trace'] for c in result['cases'] for t in c['turns']]
assert all(a['real_model_call'] for t in turns for a in t['agents'].values())
plt.rcParams.update({'font.sans-serif':['Microsoft YaHei'], 'axes.unicode_minus':False,
                     'font.size':8.5, 'pdf.fonttype':42, 'ps.fonttype':42,
                     'axes.spines.top':False,'axes.spines.right':False,
                     'legend.frameon':False})
fig, (left,right) = plt.subplots(1,2,figsize=(6.55,3.0),gridspec_kw={'wspace':.49})
roles = ['planner','teacher','reviewer']
accepted = [sum(t['agents'][r]['accepted'] is True for t in turns) for r in roles]
left.barh(range(3),accepted,color='#0072B2',height=.5,label='通过采用门槛')
left.barh(range(3),[len(turns)-n for n in accepted],left=accepted,color='#d6dce1',height=.5,label='未通过')
left.set_yticks(range(3),['规划','教学','评审']);left.invert_yaxis()
for y,n in enumerate(accepted):left.text(len(turns)+.15,y,f'{n} / {len(turns)}',va='center',fontsize=8.5)
left.set_xlim(0,len(turns)+3.3);left.set_xticks([0,3,6,9,11]);left.set_xlabel('角色调用次数\n右侧为通过数 / 总数')
left.set_title('(a) 原始角色输出的采用情况',loc='left',fontsize=9,pad=10)
left.legend(loc='lower left',bbox_to_anchor=(-.13,-.5),ncol=2,fontsize=8,handlelength=1.2,columnspacing=.8)
progress = next(c for c in result['cases'] if c['case_id']=='tutor-C05')['turns']
correct, interaction = [], []
for t in progress:
    rows=t['trace']['memory']['after']['topics']
    rows=[x for x in rows if x['topic']==t['inputs']['topic']]
    correct.append(sum(x['correct'] for x in rows))
    interaction.append(sum(x['turns'] for x in rows))
x=list(range(1,len(progress)+1))
right.plot(x,interaction,'s--',color='#6b7881',label='交互记录',lw=1.5,ms=5)
right.plot(x,correct,'o-',color='#0072B2',label='核验正确的学生作答',lw=1.8,ms=5)
right.set_xticks(x,['第1轮','第2轮','第3轮','第4轮']);right.set_ylim(0,4.6)
right.set_ylabel('累计次数');right.yaxis.set_major_locator(MaxNLocator(integer=True))
right.set_title('(b) C05：未答不增加正确次数',loc='left',fontsize=9,pad=10)
right.grid(axis='y',alpha=.16);right.legend(loc='lower left',bbox_to_anchor=(-.13,-.5),fontsize=8)
right.annotate('本轮未提供预测',xy=(4,3),xytext=(2.2,1.1),arrowprops={'arrowstyle':'->','color':'#364E5B'},fontsize=8)
fig.subplots_adjust(left=.065,right=.98,bottom=.34,top=.85)
for extension in ('pdf','png'):
    fig.savefig(OUT/f'agents-evidence.{extension}',dpi=300,bbox_inches='tight',facecolor='white')
plt.close(fig)
