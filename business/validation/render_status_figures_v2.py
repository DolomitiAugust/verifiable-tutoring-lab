"""business evidence/status revision. Preserves historical scripts and figures.

Native vector diagrams, no fabricated measurements. Run preview first, inspect
colour and grayscale, then use --export-reviewed. `--service-only` is a separate
scope added after the initial two-figure task.
"""
from pathlib import Path
import argparse
import hashlib
import json
import warnings
import importlib.util
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from PIL import Image

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
OUT=ROOT/'report/figures'
BLUE,ORANGE,GREEN='#0072B2','#E69F00','#009E73'
INK,GREY,LIGHT='#263238','#66737A','#D5DBDE'
TEXTS=[]
SOURCES={}
plt.rcParams.update({'font.sans-serif':['Microsoft YaHei'],'font.size':9,
                     'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none',
                     'axes.unicode_minus':False})

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(rel):
    p=ROOT/rel; SOURCES[rel]=sha(p)
    return json.loads(p.read_text(encoding='utf8'))
def canvas(height,ytop):
    TEXTS.clear()
    f,a=plt.subplots(figsize=(6.3,height));a.set(xlim=(0,10),ylim=(0,ytop));a.axis('off')
    f.subplots_adjust(left=.018,right=.982,bottom=.025,top=.98)
    return f,a
def txt(a,x,y,s,size=9,color=INK,bold=False,ha='center',bounds=None):
    t=a.text(x,y,s,fontsize=size,color=color,fontweight='bold' if bold else 'normal',
             ha=ha,va='center',linespacing=1.5)
    TEXTS.append((t,bounds));return t
def frame(a,x,y,w,h,color=BLUE,dashed=False):
    a.add_patch(Rectangle((x,y),w,h,facecolor='white',edgecolor=color,lw=.95,
                          linestyle=(0,(4,2.8)) if dashed else '-'))
    return (x,y,w,h)
def box(a,x,y,w,h,title,body,color=BLUE,dashed=False,body_size=8.35,title_size=9.0):
    b=frame(a,x,y,w,h,color,dashed)
    txt(a,x+w/2,y+h*.76,title,title_size,bold=True,bounds=b)
    txt(a,x+w/2,y+h*.33,body,body_size,bounds=b)
def arrow(a,p,q,dashed=False,color=GREY):
    a.add_patch(FancyArrowPatch(p,q,arrowstyle='-|>',mutation_scale=8.5,lw=.9,
                                color=color,linestyle=(0,(3,2.5)) if dashed else '-',shrinkA=0,shrinkB=0))
def evidence_map():
    synthesis=read('data/legacy/stats.json');tutor=read('tutor/legacy/results/combined.json');training=read('training/legacy/evaluation/metrics.json')
    reg=read('tutor/core/results/tool_regression_v2/metrics.json')
    new=read('tutor/core/results/new_sft_multi_paired/metrics.json')
    assert reg['automatic_passed']==reg['automatic_total']==209
    assert new['case_count']==6 and new['final_llm_used']==0
    n=[synthesis['decisions_by_split'][s]['selected'] for s in ['sft','preference','evaluation']]
    totals=tutor['totals'];assert [totals[k] for k in ['case_count','turn_count','recorded_real_model_calls','llm_final_used']]==[12,17,51,0]
    nll=[training['stages'][s]['reference_nll']['overall']['mean_nll_per_answer_token'] for s in ['base','sft','dpo']]
    f,a=canvas(4.50,7.2)
    txt(a,5,6.91,'历史实测、回归修订与商业证据边界',11,bold=True)
    for x,title in [(1.56,'历史实测（见批次标签）'),(4.87,'支持的局部结论'),(8.15,'仍需独立验证')]:
        txt(a,x,6.35,title,9.3,bold=True)
        a.plot([x-1.32,x+1.32],[6.10,6.10],color=LIGHT,lw=.8)
    ys=[4.78,3.31,1.84]
    rows=[('synthesis 原版数据与筛选',f'{n[0]}条SFT / {n[1]}对偏好\n{n[2]}条原版留出示范',
           '数据与标签可回查','核验参考输出\n检验该版筛选后的分布'),
          ('training 原CPU：SFT / DPO','128条SFT / 64对偏好\n同60题参考NLL配对',
           '仅该组参考拟合',f'{nll[0]:.3f} / {nll[1]:.3f} / {nll[2]:.3f}\nBase / SFT / DPO，生成不可靠'),
          ('tutor 原版第一轮','12场景 / 17轮\n51次角色调用，0次采用',
           '第一轮：局部流程可运行','基础记忆已有实测\n第一轮 C08 / S02 失败')]
    for y,(lt,lb,mt,mb) in zip(ys,rows):
        box(a,.1,y,2.92,1.13,lt,lb)
        arrow(a,(3.10,y+.56),(3.44,y+.56),color=BLUE)
        box(a,3.53,y,2.68,1.13,mt,mb,title_size=8.6,body_size=7.9)
    a.plot([6.61,6.61],[1.74,5.95],color=LIGHT,lw=.8)
    for y,title,body in zip(ys,['学习者收益','教师净工作量','机构交付与购买'],
        ['关闭工具后的迁移\n延迟保持与真实课程试点','备题、复核、纠错总工时\n不能只计回答速度','权限、可靠性与服务成本\n签约、回款与续费']):
        box(a,6.81,y,2.87,1.13,title,body,color=ORANGE,dashed=True)
    txt(a,5,1.26,'第二轮C08/S02回归已做；模型仍主要回退',9,bold=True)
    txt(a,5,.76,'原批次规模与失败作为历史证据保留，不代表最新批次汇总。',8.3,color=GREY)
    txt(a,5,.39,'外部研究用于设计和比较；右侧收益不能由左侧结果直接推出。',8.3,color=GREY)
    return f,a,'business-evidence-map-v2'

def validation_path():
    read('tutor/core/results/tool_regression_v2/metrics.json')
    f,a=canvas(5.15,8.6)
    txt(a,5,8.33,'验证路径：工程回归、教师盲审与真实续费',11,bold=True)
    txt(a,5,7.91,'已知故障已修复并回归；教师盲审及后续阶段尚未执行',8.5,color=GREY)
    b=frame(a,.12,5.98,3.70,1.56,LIGHT)
    txt(a,1.97,7.32,'01 工程回归与教师盲审',9.2,bold=True,bounds=b)
    done=frame(a,.25,6.65,3.44,.39,GREEN)
    txt(a,1.97,6.845,'已做：C08/S02修复与回归',8.3,bounds=done)
    pending=frame(a,.25,6.14,3.44,.39,ORANGE,True)
    txt(a,1.97,6.335,'待做：独立题集教师盲审',8.3,bounds=pending)
    arrow(a,(3.91,6.76),(4.33,6.76),True)
    box(a,4.43,5.98,5.03,1.56,'拟议门槛：工程与内容放行',
        '严重错误出现即暂停，零例不等于零风险\n工程回归不代表产品内容质量已通过',color=ORANGE,dashed=True,body_size=8.4)
    stages=[(4.43,'02 自愿教学试点','课程内对照与延迟无工具后测\n记录教师全部新增工作量',
             '拟议门槛：学习与工作量','区间跨越预设门槛：结论不确定，继续验证'),
            (2.88,'03 签约、交付与回款','核对合同、到账与服务成本\n保留拒绝和延期记录',
             '拟议门槛：买方真实支付','意向不计成交；负贡献需调价或缩小范围'),
            (1.33,'04 续费与可复制交付','再次实际付费、续期成本\n记录流失与课程适配负担',
             '拟议门槛：持续商业条件','未续费或交付不可复制：重评产品路线')]
    for y,stage,body,gate,condition in stages:
        box(a,.12,y,3.70,1.13,stage,body,color=BLUE,dashed=True)
        arrow(a,(3.91,y+.56),(4.33,y+.56),True)
        box(a,4.43,y,5.03,1.13,gate,condition,color=ORANGE,dashed=True,body_size=8.1)
    for top,bottom in [(5.98,5.56),(4.43,4.01),(2.88,2.46)]:
        arrow(a,(1.97,top-.05),(1.97,bottom+.05),True,color=BLUE)
        txt(a,2.58,(top+bottom)/2,'通过后',7.8,color=GREY)
    for y in [6.76,4.99,3.44,1.89]:a.plot([9.54,9.78],[y,y],color=ORANGE,lw=.8,ls=(0,(3,2.5)))
    a.plot([9.78,9.78],[1.89,6.76],color=ORANGE,lw=.8,ls=(0,(3,2.5)))
    arrow(a,(9.78,1.89),(9.78,.61),True,color=ORANGE)
    arrow(a,(9.78,.61),(9.46,.61),True,color=ORANGE)
    b=frame(a,.12,.26,9.34,.68,ORANGE,True)
    txt(a,4.79,.60,'任一门槛证据不足：暂停推进，返回修订；必要时停止产品化。',8.8,bounds=b)
    return f,a,'business-validation-path-v2'

def service_flow():
    # Reuse the preserved original layout without executing its entry point.
    p=ROOT/'business/render_explanatory_figures.py';SOURCES[str(p.relative_to(ROOT))]=sha(p)
    spec=importlib.util.spec_from_file_location('historical_business_layout',p)
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    old.ORANGE=ORANGE
    captured=[]
    old.save=lambda fig,name:captured.append(fig)
    old.service_flow()
    f=captured[0];a=f.axes[0];TEXTS.clear()
    found=0
    for t in a.texts:
        if t.get_text()=='保存与检索已有实测\n正确记账仍需修复':
            t.set_text('已做空预测修复与回归\n其余记账边界待扩展验证');t.set_fontsize(8.0);found+=1
        TEXTS.append((t,None))
    assert found==1
    read('tutor/core/results/tool_regression_v2/metrics.json')
    return f,a,'business-service-flow-v2'

def export(f,a,name,reviewed):
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter('always');f.canvas.draw()
    bad=[str(w.message) for w in ws if 'Glyph' in str(w.message)]
    ren=f.canvas.get_renderer();inv=a.transData.inverted()
    for t,box in TEXTS:
        b=t.get_window_extent(ren)
        if b.x0<0 or b.y0<0 or b.x1>f.bbox.x1 or b.y1>f.bbox.y1:bad.append('canvas: '+t.get_text())
        if box:
            x,y,w,h=box;b=b.transformed(inv)
            if b.x0<x-.025 or b.x1>x+w+.025 or b.y0<y-.025 or b.y1>y+h+.025:bad.append('module: '+t.get_text())
    p=HERE/(name+'-preview.png');f.savefig(p,dpi=300,facecolor='white')
    Image.open(p).convert('L').save(HERE/(name+'-grayscale.png'),dpi=(300,300))
    info={'source_sha256':SOURCES,'script_sha256':sha(Path(__file__)),'size_inches':list(f.get_size_inches()),
          'minimum_font_pt':min(t.get_fontsize() for t,_ in TEXTS),'text_count':len(TEXTS),
          'layout_findings':bad,'colours':[BLUE,ORANGE,GREEN],'reviewed_export':reviewed}
    if bad:raise ValueError(bad)
    if reviewed:
        for ext in ['pdf','png']:
            f.savefig(OUT/(name+'.'+ext),dpi=300,facecolor='white')
        info['outputs']={str((OUT/(name+'.'+e)).relative_to(ROOT)):sha(OUT/(name+'.'+e)) for e in ['pdf','png']}
    (HERE/(name+'-manifest.json')).write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf8')
    plt.close(f);return {'name':name,'layout_findings':bad,'reviewed_export':reviewed}

def main():
    p=argparse.ArgumentParser();p.add_argument('--export-reviewed',action='store_true');p.add_argument('--service-only',action='store_true');args=p.parse_args()
    for rel in ['business/render_explanatory_figures.py','report/figures/business-evidence-map.pdf','report/figures/business-validation-path.pdf','report/figures/business-service-flow.pdf']:
        SOURCES[rel]=sha(ROOT/rel)
    for make in ([service_flow] if args.service_only else [evidence_map,validation_path]):
        print(json.dumps(export(*make(),args.export_reviewed),ensure_ascii=True))
if __name__=='__main__':main()
