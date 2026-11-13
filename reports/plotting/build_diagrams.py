"""Source-bound bilingual research diagrams. No model execution or new experiment.

The palette is color-blind-friendly blue, orange, and green. All icons are editable vector metaphors.
Numerical content is loaded and checked against existing frozen project files.
"""
from __future__ import annotations
from pathlib import Path
from datetime import datetime, timezone
import argparse, hashlib, json, os, re, textwrap, warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle, Polygon, FancyArrowPatch, Arc
from matplotlib.font_manager import FontProperties
from matplotlib.text import Text
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'reports/assets'
BLUE,ORANGE,GREEN='#0072B2','#E69F00','#009E73'
INK,GREY,LIGHT='#263746','#62727C','#CCD5DA'
WIDTH_MM=162.0
SIZES={'synthesis-atlas':125.0,'teaching-memory':125.0,'evidence-to-pilot':116.0,'case-outcomes':75.0}
SOURCES={}

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(rel):
    path=ROOT/rel;SOURCES[rel]=sha(path)
    return json.loads(path.read_text(encoding='utf8'))

def evidence():
    s=read('synthesis/dataset/stats.json')
    valid=read('synthesis/validation.json')
    assert valid['passed'] and not valid['errors'] and s['candidate_count']==8240
    counts={}
    for name,n in [('sft',1000),('preference',500),('dev',36),('evaluation',60),('preference_eval',100),('independent_test',24)]:
        p=ROOT/'synthesis/dataset'/f'{name}.jsonl'
        counts[name]=sum(bool(x.strip()) for x in p.read_text(encoding='utf8').splitlines())
        assert counts[name]==n;SOURCES[str(p.relative_to(ROOT))]=sha(p)
    for stage,n in [('sft',3000),('dpo',500),('continued_sft',500)]:
        first=read(f'training/imports/run_5bf9f4d325f1/results/runs/{stage}/summary.json')
        retry=read(f'training/retry_imports/retry_048f9a92c0ef_numeric_review/retry_results/runs/{stage}/summary.json')
        assert first['steps']==retry['steps']==n
    tool=read('tutor/core/results/tool_regression_v2/metrics.json')
    assert tool['automatic_passed']==tool['automatic_total']==209 and tool['real_model_calls']==0
    linked=read('tutor/core/reviews/retry_link_summary.json')
    groups=[linked['groups'][k]['summary'] for k in ['original_retry','clean_first','clean_retry']]
    calls=sum(x['real_model_calls'] for x in groups);turns=sum(x['turn_count'] for x in groups)
    adopted=sum(x['final_llm_used'] for x in groups)
    assert (calls,turns,adopted)==(54,18,0)
    core=read('tutor/legacy/results/core/results.json');supp=read('tutor/legacy/results/supplement/results.json')
    reviews=[read('tutor/legacy/results/core/review.json'),read('tutor/legacy/results/supplement/review.json')]
    cases={c['case_id']:c for d in [core,supp] for c in d['cases']}
    scored={c['case_id']:c for d in reviews for c in d['cases']}
    for cid in ['tutor-C03','tutor-C04','tutor-C08','tutor-S02']:
        expected=cases[cid]['expected_before_run']['completion']
        assert expected==('partial' if cid in ['tutor-C03','tutor-C04'] else 'complete')
        complete=all(x['score']==2 for x in scored[cid]['criteria'])
        assert complete==(cid in ['tutor-C03','tutor-C04'])
        assert all(t['trace']['final']['source']=='verified_template' and not t['trace']['final']['llm_used'] for t in cases[cid]['turns'])
    empty=cases['tutor-S02']['turns'][0]['trace']
    assert empty['input']['student_answer']==empty['tool']['main']['output']==''
    assert empty['student_evidence']['status']=='unanswered'
    stress=read('business/validation/results/quality_cost_stress.json')
    assumptions=read('business/business_assumptions.json')
    assert stress['review_rates_are_measured'] is False
    assert stress['rule_fallback_is_not_equivalent_to_human_review'] is True
    rows=stress['rows'];base=next(x for x in rows if x['review_fraction']==.02)
    y0=rows[0]['contribution_after_incremental_cac_cny']
    slope=(rows[-1]['contribution_after_incremental_cac_cny']-y0)/rows[-1]['review_fraction']
    threshold=-y0/slope
    assert abs(threshold-.17028493827160494)<1e-10
    for rel in ['reports/source/en/chapters/data.tex','tutor/core/system.py','tutor/core/memory.py']:
        if (ROOT/rel).exists():SOURCES[rel]=sha(ROOT/rel)
    return dict(counts=counts,candidates=s['candidate_count'],training_families=len(s['training_families']),
                calls=calls,turns=turns,adopted=adopted,tool_checks=209,threshold=threshold,
                review_assumption=.02,contribution_at_zero=y0,contribution_slope=slope,
                cost_rows=rows,base_cost=base,case_ids=['C03','C04','C08','S02'])

class Drawing:
    def __init__(self,name,lang):
        self.name,self.lang=name,lang;self.height=SIZES[name]/WIDTH_MM*100
        self.fig=plt.figure(figsize=(WIDTH_MM/25.4,SIZES[name]/25.4),facecolor='white')
        self.ax=self.fig.add_axes([0,0,1,1]);self.ax.set(xlim=(0,100),ylim=(0,self.height));self.ax.axis('off')
        self.font='DejaVu Sans' if lang=='en' else 'Microsoft YaHei'
        self.texts=[];self.bounds=[];self.arrow_records=[]
    def tr(self,en,zh):return en if self.lang=='en' else zh
    def label(self,x,y,en,zh=None,size=8.2,bold=False,color=INK,width=94,height=None,ha='center',va='center'):
        value=en if self.lang=='en' or zh is None else zh
        text=self.ax.text(x,y,value,fontsize=size,fontfamily=self.font,fontweight='bold' if bold else 'normal',
                          color=color,ha=ha,va=va,linespacing=1.15,zorder=5)
        self.fit(text,value,width,height,minsize=7.35 if size>=7.35 else size)
        self.texts.append(text);return text
    def fit(self,obj,value,width,height,minsize):
        renderer=self.fig.canvas.get_renderer()
        px_width=width/100*self.fig.bbox.width
        px_height=height/100*self.fig.bbox.width if height else None
        def wrap_line(line):
            words=re.findall(r'[\u3400-\u9fff]|[^\s\u3400-\u9fff]+|\s+',line)
            answer=[];current=''
            for word in words:
                test=current+word
                tw=renderer.get_text_width_height_descent(test,obj.get_fontproperties(),False)[0]
                if current and tw>px_width:answer.append(current.rstrip());current=word.lstrip()
                else:current=test
            answer.append(current.rstrip());return answer
        for size in np.arange(obj.get_fontsize(),minsize-.01,-.15):
            obj.set_fontsize(max(float(size),minsize))
            obj.set_text('\n'.join(part for line in value.split('\n') for part in wrap_line(line)))
            bb=obj.get_window_extent(renderer)
            if bb.width<=px_width+.5 and (px_height is None or bb.height<=px_height+.5):return
    def frame(self,x,y,w,h,color=LIGHT,dashed=False,rounded=False):
        patch=(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=.8',facecolor='white',
                              edgecolor=color,lw=.8,linestyle=(0,(3,2)) if dashed else '-') if rounded else
               Rectangle((x,y),w,h,facecolor='white',edgecolor=color,lw=.8,linestyle=(0,(3,2)) if dashed else '-'))
        self.ax.add_patch(patch);return patch
    def rule(self,x1,x2,y,color=LIGHT):self.ax.plot([x1,x2],[y,y],color=color,lw=.7,zorder=1)
    def arrow(self,a,b,color=GREY,dashed=False,rad=0):
        self.ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=8,linewidth=.85,color=color,
                          linestyle=(0,(3,2)) if dashed else '-',connectionstyle=f'arc3,rad={rad}',
                          shrinkA=0,shrinkB=0,zorder=2))
        self.arrow_records.append(dict(start=list(a),end=list(b),kind='proposed/control' if dashed else 'data/evidence',color=color))
    def title(self,en,zh,subtitle_en,subtitle_zh):
        self.label(50,self.height-3,en,zh,size=11.2,bold=True,width=94,height=4.2)
        self.label(50,self.height-6.8,subtitle_en,subtitle_zh,size=7.9,color=GREY,width=94,height=3.3)
    def section(self,x,y,w,en,zh,color=BLUE):
        self.label(x,y,en,zh,size=8.0,bold=True,color=color,ha='left',width=w,height=3)
        self.rule(x,x+w,y-2.1)
    def robot(self,x,y,color=BLUE,role='planner',scale=1):
        a=self.ax;s=scale
        a.add_patch(Circle((x,y+1.8*s),1.8*s,facecolor='white',edgecolor=color,lw=.8))
        a.add_patch(FancyBboxPatch((x-1.7*s,y-2.7*s),3.4*s,2.8*s,boxstyle='round,pad=0,rounding_size=.5',facecolor='white',edgecolor=color,lw=.8))
        for ex in [-.65,.65]:a.add_patch(Circle((x+ex*s,y+2.05*s),.15*s,facecolor=INK,edgecolor='none'))
        a.add_patch(Arc((x,y+1.7*s),1.1*s,.7*s,theta1=205,theta2=335,edgecolor=INK,lw=.65))
        for side in [-1,1]:a.plot([x+side*1.7*s,x+side*2.4*s],[y-1.0*s,y-1.7*s],color=color,lw=.8)
        if role=='planner':
            a.add_patch(Rectangle((x+1.1*s,y-2.3*s),1.7*s,2.5*s,facecolor='white',edgecolor=GREY,lw=.6))
            for yy in [-1.7,-1.1,-.5]:a.plot([x+1.45*s,x+2.4*s],[y+yy*s]*2,color=GREY,lw=.5)
        elif role=='teacher':
            a.add_patch(Polygon([(x-1.2*s,y-.8*s),(x,y-1.2*s),(x+1.2*s,y-.8*s),(x+1.2*s,y-2.4*s),(x,y-2.8*s),(x-1.2*s,y-2.4*s)],facecolor='white',edgecolor=color,lw=.6))
        else:
            a.add_patch(Circle((x+2.3*s,y-.7*s),.85*s,facecolor='white',edgecolor=color,lw=.7))
            a.plot([x+2.8*s,x+3.5*s],[y-1.3*s,y-2.1*s],color=color,lw=.8)
    def factory(self,x,y):
        a=self.ax
        a.add_patch(Polygon([(x,y),(x+11,y),(x+11,y+7),(x+8,y+5),(x+8,y+7),(x+5,y+5),(x+5,y+7),(x,y+4)],facecolor='white',edgecolor=BLUE,lw=.9))
        a.add_patch(Rectangle((x+.8,y+5),1.5,5,facecolor='white',edgecolor=BLUE,lw=.8))
        for xx in [2,4.8,7.6]:a.add_patch(Rectangle((x+xx,y+1.4),1.4,1.8,facecolor='white',edgecolor=GREY,lw=.6))
        a.plot([x-.5,x+11.6],[y-1.0,y-1.0],color=GREY,lw=.7)
        for xx in [1.0,4.5,8]:a.add_patch(Circle((x+xx,y-1.8),.6,facecolor='white',edgecolor=GREY,lw=.6))
    def sheets(self,x,y,color=BLUE):
        for off in [1.0,.5,0]:self.ax.add_patch(Rectangle((x+off,y+off),5,7,facecolor='white',edgecolor=color,lw=.65))
        for yy in [1.4,3,4.6,5.8]:self.ax.plot([x+1,x+4],[y+yy,y+yy],color=GREY,lw=.5)
    def queue(self,x,y,color=ORANGE):
        for xx in [0,4.1,8.2]:
            self.frame(x+xx,y,3.0,4.0,color)
            self.rule(x+xx+.6,x+xx+2.4,y+2.6,GREY)
            self.rule(x+xx+.6,x+xx+2.4,y+1.4,GREY)
        self.arrow((x+3.1,y+2),(x+4,y+2),color)
        self.arrow((x+7.2,y+2),(x+8.1,y+2),color)
    def drawers(self,x,y,w,h):
        self.frame(x,y,w,h,GREEN)
        for yy in [y+h*.2,y+h*.6]:
            self.frame(x+1,yy,w-2,h*.25,LIGHT)
            self.rule(x+w*.42,x+w*.58,yy+h*.14,GREEN)

def synthesis(lang,data):
    d=Drawing('synthesis-atlas',lang);H=d.height
    d.title('From research ideas to frozen teaching data','从研究借鉴到冻结教学数据',
            'Selection and execution are different design questions','“选什么”与“怎么运行”是两个不同问题')
    d.section(3,66,45,'A  WHAT TO SELECT','A  选择什么样的内容')
    d.section(52,66,45,'B  HOW ITEMS ADVANCE','B  样本如何逐项推进',ORANGE)
    d.sheets(5.5,53.0,BLUE);d.queue(54.5,55,ORANGE)
    d.label(29,61.5,'NaturalThoughts','NaturalThoughts',9.0,bold=True,width=34,height=3.5)
    d.label(29,56.5,'Annotate reasoning traces\n→ select → distill','标注推理轨迹\n→ 选择 → 蒸馏',8.2,width=34,height=7)
    d.label(29,51.8,'Strategies, verbosity, difficulty proxies','策略、冗长程度与难度代理',7.6,color=GREY,width=40,height=4)
    d.label(80,61.5,'Matrix','Matrix',9.0,bold=True,width=29,height=3.5)
    d.label(80,56.5,'Stateful items + role queues\nShared model / tool services','样本状态 + 角色队列\n共享模型与工具服务',8.2,width=30,height=7)
    d.label(75,51.8,'Scheduling is not a truth criterion','调度机制不等于事实核验',7.6,color=GREY,width=42,height=4)
    # Dashed transfer denotes selected design ideas, not a reproduced framework.
    d.ax.plot([26,26,50,75,75],[49.5,47.5,47.5,47.5,49.5],color=GREY,lw=.7,ls=(0,(3,2)))
    d.arrow((50,47.5),(50,44.5),GREY,True)
    d.label(59,46,'Adapt ideas only','只借鉴设计思想',7.4,color=GREY,width=20,height=3)
    d.section(3,42.5,94,'C  LOCAL EXECUTABLE FACTORY','C  本项目：可执行数据工厂')
    d.factory(3.8,29)
    d.label(9.5,25.4,'Rules + templates','规则与模板',7.5,width=17,height=3)
    d.frame(21,28,19,11,BLUE,rounded=True)
    d.label(30.5,35.8,f"{data['candidates']:,} candidates",f"{data['candidates']:,} 条候选",8.5,bold=True,width=17,height=3.2)
    d.label(30.5,31.2,'Code, requests,\nresponse pairs','程序、请求\n与回答对',7.7,width=17,height=5.2)
    d.robot(48,33,ORANGE,'reviewer',1.1)
    d.label(49.5,25.4,'Execute + verify','执行与独立核验',7.5,width=21,height=3)
    d.frame(61,28,16,11,BLUE,rounded=True)
    d.label(69,35.8,'Filter','筛选',8.6,bold=True,width=14,height=3)
    d.label(69,31.2,'Facts, task, length,\nprogram overlap','事实、任务、长度\n程序重合',7.7,width=14,height=5.2)
    d.frame(82,28,15,11,GREEN,rounded=True)
    d.label(89.5,35.8,'Freeze splits','冻结各集合',8.3,bold=True,width=13,height=3.2)
    d.label(89.5,31.2,'Hash, seed,\nexport ledger','哈希、种子\n与导出台账',7.7,width=13,height=5.2)
    for a,b in [((15.5,33),(20,33)),((40.8,33),(44,33)),((53,33),(60,33)),((77.8,33),(81,33))]:d.arrow(a,b,BLUE)
    d.section(3,21.5,94,'D  FROZEN PURPOSES — TESTS DO NOT TRAIN','D  用途预先冻结：测试不进入训练')
    d.frame(3,8,26,10,BLUE);d.frame(33,8,19,10,ORANGE);d.frame(56,8,41,10,GREEN)
    d.label(16,15.5,'Train only','只用于训练',8.3,bold=True,width=24,height=3)
    d.label(16,11.2,'1,000 SFT / 500 pairs','1,000 条 SFT / 500 对偏好',7.8,width=24,height=4)
    d.label(42.5,15.5,'Select epoch','选择轮次',8.3,bold=True,width=17,height=3)
    d.label(42.5,11.2,'36 dev','36 条开发题',8.1,width=17,height=4)
    d.label(76.5,15.5,'Evaluate only','只用于最终评估',8.3,bold=True,width=39,height=3)
    d.label(76.5,11.2,'60 bridge / 100 preferences / 24 natural','60 条桥接 / 100 对偏好 / 24 条自然题',7.8,width=39,height=4)
    d.label(50,3.3,'No DeepSeek-teacher distillation or distributed cluster was run here. Icons are metaphors.',
            '本项目未运行 DeepSeek 教师蒸馏或分布式集群；小图标仅为帮助理解的比喻。',7.4,color=GREY,width=94,height=4)
    return d

def teaching(lang,data):
    d=Drawing('teaching-memory',lang)
    d.title('Teaching calls, evidence gates, and memory','教学角色、证据门与记忆',
            'A model draft, a verified answer, and a learner record are different objects','模型草稿、核验答案与学习记录是三个不同对象')
    d.section(3,66,94,'A  SHARED WEIGHTS WITHIN EACH CONDITION','A  每个条件内共享同一组权重')
    d.ax.plot([14,58],[62.5,62.5],color=BLUE,lw=.8)
    for x,color,role,en,zh in [(14,BLUE,'planner','Planner','规划'),(36,ORANGE,'teacher','Teacher','教学'),(58,GREEN,'reviewer','Reviewer','评审')]:
        d.arrow((x,62.5),(x,60.5),BLUE,True)
        d.robot(x,56.5,color,role,1.2)
        d.label(x,50.1,en,zh,8.3,bold=True,width=18,height=3.1)
    for a,b in [((18,56),(29,56)),((40,56),(51,56)),((64,56),(72,56))]:d.arrow(a,b,BLUE)
    d.frame(73,50,24,12,BLUE,rounded=True)
    d.label(85,58.7,'Content gate','独立内容门',8.5,bold=True,width=21,height=3.3)
    d.label(85,53.7,'Facts / constraints / review','事实 / 请求约束 / 评审',7.6,width=21,height=5.2)
    d.frame(3,43,63,4.7,LIGHT)
    d.label(34.5,45.35,'Shared context: verified facts + retrieved learner history','共享上下文：已核验事实 + 检索到的学习记录',7.8,width=60,height=3.5)
    for x in [14,36,58]:d.arrow((x,47.7),(x,48.4),GREY)
    d.frame(73,42.8,10,5,GREEN);d.frame(86,42.8,11,5,ORANGE)
    d.label(78,45.3,'Adopt','采用',7.7,width=8,height=3)
    d.label(91.5,45.3,'Fallback','明确回退',7.4,width=9,height=3)
    d.arrow((79,49.7),(78,48),GREEN)
    d.arrow((90.5,49.7),(91.5,48),ORANGE,True)
    d.frame(3,31.6,25,8.5,BLUE)
    d.label(15.5,37.7,'Restricted execution','受限程序核验',8.1,bold=True,width=22,height=3)
    d.label(15.5,34.0,'Code + student prediction','程序 + 学生的实际预测',7.6,width=22,height=3)
    d.arrow((15.5,40.3),(15.5,42.7),BLUE)
    d.frame(35,31.6,31,8.5,GREEN)
    d.drawers(36,32.5,4.8,6.4)
    d.label(53.8,37.9,'SQLite: learner + topic','SQLite：学生 + 知识点',7.8,bold=True,width=23.2,height=3.8)
    d.label(53.8,34.1,'Raw events + counters','原始事件 + 分组计数',7.5,width=23.2,height=3.2)
    d.arrow((50.5,40.3),(50.5,42.7),GREEN)
    d.arrow((28.3,35.2),(34.6,35.2),GREEN)
    d.label(31.4,38.7,'Store','记账',7.4,color=GREY,width=6,height=3)
    d.label(84.5,36.4,'Separate provenance:\nmodel text or tool/template','分别标记来源：\n模型文字或工具/模板',7.7,color=GREY,width=26,height=7)
    d.label(50,28.2,f"Three added linked groups: {data['calls']} calls / {data['turns']} turns; {data['adopted']}/{data['turns']} final model prose adopted.",
            f"三组新增联动：{data['calls']} 次调用 / {data['turns']} 轮；最终采用模型文字 {data['adopted']}/{data['turns']}。",7.8,color=GREY,width=94,height=3.3)
    d.section(3,24.2,94,'B  PRESENCE IS NOT STRING TRUTHINESS — HISTORICAL S02','B  “是否提交”不等于“字符串是否非空”：历史 S02',ORANGE)
    for x,w,c in [(3,26,BLUE),(35,28,ORANGE),(69,28,ORANGE)]:d.frame(x,10.0,w,11.5,c)
    d.label(16,18.4,'Input semantics','输入语义',8.0,bold=True,width=23,height=3)
    d.label(16,13.7,'None = no submission\n"" = submitted empty string','None = 尚未提交\n"" = 已提交的空字符串',7.6,width=23,height=6)
    d.label(49,18.4,'Old check: truthiness','旧判断：检查非空',8.0,bold=True,width=25,height=3)
    d.label(49,13.7,'bool("".strip()) = False\n→ marked unanswered','bool("".strip()) = False\n→ 被标成未作答',7.6,width=25,height=6)
    d.label(83,18.4,'Feedback → memory error','反馈错误 → 记账错误',8.0,bold=True,width=25,height=3)
    d.label(83,13.7,'attempts +0 / correct +0\nunanswered +1','作答 +0 / 正确 +0\n未答 +1',7.6,width=25,height=6)
    d.arrow((29.5,15.5),(34.5,15.5),ORANGE);d.arrow((63.5,15.5),(68.5,15.5),ORANGE)
    d.frame(3,2.4,94,5.3,GREEN)
    d.label(50,5.05,'Revised, on this S02 case: student_answer is not None → compare empty strings\n→ attempts +1; correct +1',
            '对 S02 本例，修复后：student_answer is not None → 空串也比较 → 作答 +1；正确 +1',7.8,width=90,height=4.5)
    return d

def pilot(lang,data):
    d=Drawing('evidence-to-pilot',lang)
    d.title('From prototype evidence to a testable pilot','从原型证据走向可检验的试点',
            'Completed engineering does not establish learning gains or commercial demand','工程完成不等于学习收益或商业需求已经得到验证')
    for x,w,c,dashed in [(3,28,BLUE,False),(36,28,ORANGE,False),(69,28,ORANGE,True)]:d.frame(x,30,w,30,c,dashed)
    d.label(17,57.5,'OBSERVED','本项目实测',8.3,bold=True,color=BLUE,width=25,height=3)
    d.label(50,57.5,'CURRENT LIMITS','当前证据边界',8.3,bold=True,color=ORANGE,width=25,height=3)
    d.label(83,57.5,'PROPOSED GATES','拟议验证门',8.3,bold=True,color=GREEN,width=25,height=3)
    d.label(17,51.9,'Frozen teaching data\n1,000 SFT / 500 pairs','冻结教学数据\n1,000 条 SFT / 500 对偏好',8.0,width=24,height=7)
    d.label(17,44.9,'Two cloud runs\nSFT → DPO + control','两轮云端实验\nSFT → DPO + 对照',8.0,width=24,height=7)
    d.label(17,37.7,'209/209 tool-only checks\nNot an LLM teaching score','纯工具回归 209/209\n不是模型教学得分',7.8,width=24,height=7)
    d.label(50,50.9,'3 added linked groups:\n54 calls / 18 turns;\n0/18 model-prose finals','三组新增联动：\n54 次调用 / 18 轮；\n最终模型文字采用 0/18',7.8,width=24,height=9.6)
    d.label(50,39.6,'Learning gains and\npaid demand unobserved','学习收益与付费需求\n尚未实测',8.0,width=24,height=7)
    d.label(83,52.5,'Teacher blind review','教师盲审',8.0,bold=True,width=24,height=3)
    d.label(83,48.6,'Facts + requested constraints','事实与请求约束',7.6,width=24,height=3.4)
    d.label(83,43.7,'Voluntary course trial','自愿课程试点',8.0,bold=True,width=24,height=3)
    d.label(83,40.0,'Unaided transfer + total work','无工具迁移与总工作量',7.6,width=24,height=3.4)
    d.label(83,36.0,'Paid pilot + renewal','付费试点与续费',8.0,bold=True,width=24,height=3)
    d.label(83,32.4,'Costs + actual receipts','成本与真实回款',7.6,width=24,height=3.4)
    d.arrow((31.6,46),(35.4,46),GREY);d.arrow((64.6,46),(68.4,46),ORANGE,True)
    d.section(3,28.1,94,'COST INSET — DETERMINISTIC BUSINESS ASSUMPTIONS','成本缩略图：确定性商业假设，不是学习性能',ORANGE)
    ax=d.fig.add_axes([.095,.092,.445,.265])
    x=np.array([0,.02,.05,.10,.20,.25])
    y=(data['contribution_at_zero']+data['contribution_slope']*x)/10000
    ax.plot(x*100,y,color=BLUE,lw=1.2,marker='o',markersize=2.4)
    ax.axhline(0,color=GREY,lw=.65,ls='--');ax.axvline(data['threshold']*100,color=ORANGE,lw=.75,ls=':')
    ax.scatter([2],[data['base_cost']['contribution_after_incremental_cac_cny']/10000],color=GREEN,s=18,marker='s',zorder=4)
    ax.scatter([data['threshold']*100],[0],color=ORANGE,s=20,marker='D',zorder=4)
    ax.set(xlim=(0,25),ylim=(-3,6),xticks=[0,5,10,15,20,25],yticks=[-2,0,2,4,6])
    ax.set_xlabel(d.tr('Human-review turn share (%)','人工复核轮次占比（%）'),fontsize=7.5,fontfamily=d.font,labelpad=2)
    ax.set_ylabel(d.tr('Contribution (10,000 CNY)','贡献（万元）'),fontsize=7.5,fontfamily=d.font,labelpad=2)
    ax.tick_params(labelsize=7.3,length=2,pad=2)
    for t in ax.get_xticklabels()+ax.get_yticklabels():t.set_fontfamily(d.font)
    ax.spines[['top','right']].set_visible(False)
    ax.spines[['left','bottom']].set_color(LIGHT);ax.grid(axis='y',color='#E8ECEF',lw=.45)
    d.frame(61,5.1,36,19.3,ORANGE,True)
    d.label(79,21.0,'2% review is assumed','2% 复核比例只是商业假设',8.0,bold=True,color=ORANGE,width=32,height=4.5)
    d.label(79,15.6,f"{data['threshold']*100:.2f}%: zero contribution\nafter CAC in this scenario",f"{data['threshold']*100:.2f}%：本成本情景下\n获客后贡献降至零",7.8,width=32,height=6.5)
    d.label(79,8.8,'Not an allowed error rate\nor a learning-quality threshold','不是允许的错误率\n也不是学习质量门槛',7.6,color=GREY,width=32,height=5.8)
    return d

def cases(lang,data):
    d=Drawing('case-outcomes',lang)
    d.title('Four cases against their frozen expectations','四个案例：对照原先冻结的预期',
            'Expected task completion → observed completion; historical results retained','预期完成程度 → 实际完成程度；保留历史结果')
    content=[(3,23,BLUE,'C03  Across-instance memory','C03  跨实例记忆','Partial → complete','部分 → 完整','Earlier error retrieved after restart;\nnew program facts checked.','重建系统后取回旧错因，\n核对新题事实。'),
             (52,23,GREEN,'C04  Student isolation','C04  学生隔离','Partial → complete','部分 → 完整','New student: no other learner’s history;\nunanswered status preserved.','新学生不读取他人记录，\n未作答状态保持正确。'),
             (3,8,ORANGE,'C08  Constrained follow-up','C08  定向变式','Complete → incomplete','完整 → 未完整','Executable output, wrong edit:\naccumulator changed instead of list.','程序可执行，但改错位置：\n改累计值，而非补齐列表。'),
             (52,8,ORANGE,'S02  Valid empty prediction','S02  有效空预测','Complete → incomplete','完整 → 未完整','Correct empty output; valid prediction\nrecorded as unanswered.','空输出正确，却把有效空预测\n记录成了未作答。')]
    for x,y,color,head,zh,state,sz,body,bz in content:
        d.frame(x,y,45,13,color,rounded=True)
        d.label(x+22.5,y+10.7,head,zh,8.35,bold=True,width=41,height=3.2)
        d.label(x+22.5,y+7.5,state,sz,8.0,bold=True,color=color,width=41,height=3)
        d.label(x+22.5,y+3.3,body,bz,7.75,width=41,height=6)
    d.label(50,4.6,'Success mechanisms: tools/templates + memory, not improved model collaboration.',
            '成功来源是工具、模板与记忆，不代表模型协作能力得到提升。',7.4,color=GREY,width=94,height=3)
    d.label(50,1.8,'C07 remains an expected limitation; no case was relabeled.',
            'C07 仍保留“预期局限”标签；没有改标签来凑案例。',7.4,color=GREY,width=94,height=3)
    return d

BUILDERS={'synthesis-atlas':synthesis,'teaching-memory':teaching,'evidence-to-pilot':pilot,'case-outcomes':cases}

def audit(d):
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always');d.fig.canvas.draw()
    findings=[str(x.message) for x in captured if 'Glyph' in str(x.message)]
    r=d.fig.canvas.get_renderer();fw,fh=d.fig.bbox.width,d.fig.bbox.height
    for text in d.fig.findobj(Text):
        if not text.get_visible() or not text.get_text().strip():continue
        b=text.get_window_extent(r)
        if b.x0<0 or b.y0<0 or b.x1>fw or b.y1>fh:findings.append('canvas overflow: '+text.get_text())
    overlaps=[]
    for i,a in enumerate(d.texts):
        ba=a.get_window_extent(r)
        for b in d.texts[i+1:]:
            bb=b.get_window_extent(r)
            if min(ba.x1,bb.x1)-max(ba.x0,bb.x0)>1 and min(ba.y1,bb.y1)-max(ba.y0,bb.y0)>1:
                overlaps.append([a.get_text(),b.get_text()])
    if overlaps:findings.extend('text overlap: '+repr(x) for x in overlaps)
    return dict(findings=findings,text_objects=len(d.texts),minimum_font_pt=min(t.get_fontsize() for t in d.texts),
                width_mm=WIDTH_MM,height_mm=SIZES[d.name],all_text_searchable=True)

def render_selected(names=None,langs=None):
    plt.rcParams.update({'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','axes.unicode_minus':False,'font.size':8,
                         'figure.dpi':150,'savefig.bbox':None,'font.family':'DejaVu Sans'})
    OUT.mkdir(exist_ok=True);data=evidence();records=[]
    previous=json.loads((OUT/'manifest.json').read_text(encoding='utf8')) if (OUT/'manifest.json').exists() else {}
    image_before={str(p.relative_to(OUT)):sha(p) for lang in ['en','zh'] for p in (OUT/lang).glob('*') if p.suffix in ['.pdf','.svg','.png']}
    for lang in langs or ['en','zh']:
        dest=OUT/lang;dest.mkdir(exist_ok=True)
        for name in names or BUILDERS:
            d=BUILDERS[name](lang,data);qa=audit(d)
            files={}
            for ext in ['png','pdf','svg']:
                path=dest/f'{name}.{ext}';temporary=path.with_name(path.name+'.tmp')
                metadata={'Creator':'Codex; Academic Figure Prompt; source-bound editable vector figure','CreationDate':None,'ModDate':None} if ext=='pdf' else None
                d.fig.savefig(temporary,format=ext,dpi=300,facecolor='white',bbox_inches=None,metadata=metadata)
                os.replace(temporary,path);files[ext]={'path':str(path.relative_to(ROOT)),'sha256':sha(path)}
            record={'basename':name,'language':lang,'source':str(Path(__file__).relative_to(ROOT)),'source_sha256':sha(__file__),
                    'sources_sha256':dict(SOURCES),'data_bindings':data,'outputs':files,'layout_qa':qa,
                    'evidence_kind':'Source-bound conceptual vector diagram; cost inset is a deterministic scenario, not a learning metric.',
                    'arrows':d.arrow_records,'ai_image_generated':False,'visual_qa':'pending',
                    'labels':[{'text':t.get_text(),'position':list(t.get_position()),'fontsize_pt':t.get_fontsize()} for t in d.texts]}
            (dest/f'{name}.provenance.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
            records.append(record);plt.close(d.fig)
    updated={(r['language'],r['basename']):r for r in previous.get('figures',[])}
    updated.update({(r['language'],r['basename']):r for r in records})
    untouched={p:h for p,h in image_before.items() if (Path(p).parts[0],Path(p).stem) not in {(r['language'],r['basename']) for r in records}}
    assert all(sha(OUT/p)==digest for p,digest in untouched.items())
    manifest=dict(previous)
    manifest.update({'created_at_utc':datetime.now(timezone.utc).isoformat(),'figures':list(updated.values()),
        'new_model_experiments':False,'original_figures_unchanged':True,'palette':[BLUE,ORANGE,GREEN],
        'latest_redraw':{'variants':[r['language']+'/'+r['basename'] for r in records],'other_image_files_byte_unchanged':len(untouched)}})
    temporary=OUT/'manifest.json.tmp';temporary.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf8');os.replace(temporary,OUT/'manifest.json')
    print(json.dumps({'figures':len(records),'findings':{r['language']+'/'+r['basename']:r['layout_qa']['findings'] for r in records}},ensure_ascii=True))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--only',nargs='+',choices=list(BUILDERS));p.add_argument('--language',choices=['en','zh']);a=p.parse_args()
    render_selected(a.only,[a.language] if a.language else None)
