"""24 individually authored questions, not sampled from the curriculum.

Same Codex author, different question construction; NOT human/teacher assessment.
Gold values below were written before execution and are checked independently.
Do not revise this file after observing any model output.
"""
import ast
import hashlib
import io
import json
from contextlib import redirect_stdout
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'dataset'
# (topic, code, natural request, gold, explanation). Explicit records, no template loop.
CASES=[
('assignment', 'wallet = 137\nsaved = wallet\nwallet = wallet - 18\nprint(saved)',
 '我以为saved会跟着wallet一起变小。你能告诉我实际打印什么，并解释这种理解哪里不对吗？', '137',
 'saved在更新前保存整数137；后面对wallet重新赋值不会改动saved。'),
('split_join', "label = 'oak|pine|elm|ash'\nparts = label.split('|')\nprint('/'.join(parts[1:3]))",
 '我在处理树木标签。这里只截了哪些名字？最后连起来的文本是什么？请直接用这个例子解释。', 'pine/elm',
 '切片1:3保留下标1、2的pine和elm，join在二者之间插入斜杠。'),
('zip', 'a = [101, 103, 107]\nb = [2, 4]\nprint(sum(x - y for x, y in zip(a, b)))',
 '第三个数107去哪了？请解释配对为什么结束，并给出最后的输出。', '198',
 'zip在较短列表用完后结束，只计算101-2和103-4，相加得到198。'),
('alias', 'first = [131, 139]\nsecond = first\nfirst.append(149)\nprint(len(second))',
 '我没有往second里加东西，为什么它的长度可能改变？请用这段代码说明，并给出输出。', '3',
 '两个变量指向同一个列表；通过first追加的149也能通过second看到，长度为3。'),
('copy', 'first = [151, 157]\nsecond = first[:]\nfirst.append(163)\nprint(len(second))',
 '和直接写second = first相比，这里的切片有什么作用？最后到底打印几？', '2',
 '切片创建了新列表；first的追加不影响second，因此它仍有两项。'),
('range', 'total = 0\nfor n in range(101, 111, 3):\n    total += n\nprint(total)',
 '我不确定111要不要加进去。请列出真正参与累加的数，并给出结果。', '422',
 '参与的是101、104、107、110；range不包含终点111，四项之和是422。'),
('negative_range', 'total = 0\nfor n in range(118, 107, -4):\n    total += n\nprint(total)',
 '步长是负数时我总会漏一项。这里会跑哪几轮，最终输出多少？', '342',
 'n依次为118、114、110，下一项106已经越过终点107，因此总和为342。'),
('boundary', 'score = 143\nif score > 143:\n    bonus = 12\nelse:\n    bonus = 7\nprint(bonus)',
 '分数刚好等于门槛时会进入哪个分支？请解释并给出输出。', '7',
 '143 > 143为假，因为>不包括相等；程序进入else，输出7。'),
('division', 'print(137 // 12)',
 '不用讲解，我只想核对这行的输出值。', '11',
 ''),
('modulo', 'print(137 % 12)',
 '这里我应该报完整组数还是剩下的数量？请结合137和12说明。', '5',
 '137=12×11+5，%返回剩余数量5。'),
('slice', "word = 'strawberry'\nprint(word[1:8:2])",
 '我把步长理解成了“跳过两个”。请纠正这个理解，并写出实际输出。', 'tabr',
 '步长2表示每次下标增加2，访问1、3、5、7，对应t、a、b、r。'),
('index', 'values = [173, 179, 181, 191]\nprint(values[-2])',
 '负下标从哪里开始数？这段程序拿的是哪一项，输出是什么？', '181',
 '-1是最后一项191，-2是倒数第二项181。'),
('comprehension', 'values = [12, 15, 18, 21]\nprint([x + 2 for x in values if x % 3 == 0 and x > 15])',
 '这里有两个筛选条件。我能不能先把所有数字加2？请按正确顺序解释结果。', '[20, 23]',
 '先按原值检查两个条件，只有18和21通过；然后各加2，得到[20, 23]。'),
('set', 'items = [211, 223, 211, 227, 223, 211]\nprint(len(set(items)))',
 '我数了六个位置，但同学说结果不是6。谁的理解符合这段代码，为什么？', '3',
 'set去掉重复值，只剩211、223、227，len得到3。'),
('return', 'def adjust(x):\n    x += 7\n    return x * 2\nprint(adjust(123))',
 '请不要只报答案，帮我追踪参数进入函数以后发生的两步变化。', '260',
 'x先从123加7变成130，再乘2并返回260。'),
('default_arg', 'def add(x, box=[]):\n    box.append(x)\n    return len(box)\nadd(127)\nprint(add(131))',
 '我以为第二次调用时box会重新变成空列表，所以猜1。实际是什么？请说明。', '2',
 '默认列表在函数定义时创建，两次调用复用同一个列表，第二次追加后有两项。'),
('dict_update', "counts = {}\nfor key in ['red', 'blue', 'red', 'red']:\n    counts[key] = counts.get(key, 0) + 2\nprint(counts['red'])",
 '这个循环每次加的是次数1还是数值2？请只围绕red解释最终结果。', '6',
 'red出现三次，每次在已有值上加2，因此累计为6。'),
('nested', 'pairs = 0\nfor x in [2, 5, 9]:\n    for y in [3, 7]:\n        if x < y:\n            pairs += 1\nprint(pairs)',
 '我把两个列表长度相乘得到6。请指出哪些组合真正通过了条件，输出多少？', '3',
 '通过的是(2,3)、(2,7)、(5,7)，一共3组；另外三组不满足x<y。'),
('continue', 'total = 0\nfor x in [101, 102, 103, 104]:\n    if x % 2 == 0:\n        continue\n    total += x\nprint(total)',
 'continue是不是让整个循环结束？请借这段程序解释，并给出输出。', '204',
 'continue只跳过当前这一轮，偶数不累加；101和103相加得到204。'),
('break', 'total = 0\nfor x in [101, 102, 103, 104]:\n    if x == 103:\n        break\n    total += x\nprint(total)',
 '103会先加进总数再停止吗？请检查执行顺序并回答。', '203',
 '到103时先遇到break，后面的累加不再执行；只有101和102被加进来，得到203。'),
('recursion', 'def climb(n):\n    if n == 0:\n        return 2\n    return climb(n - 1) + 3\nprint(climb(4))',
 '我能看懂停止条件，但不会把返回值一层层算回来。请简要展示过程和最终输出。', '14',
 'climb(0)=2，往上每层加3，依次得到5、8、11、14。'),
('enumerate', "total = 0\nfor i, word in enumerate(['sun', 'moon', 'star'], start=4):\n    total += i\nprint(total)",
 'start=4是跳过前四个单词吗？请说明它真正改变了什么，以及最终结果。', '15',
 'start改变编号起点，不跳过元素；三个编号是4、5、6，总和为15。'),
('sort', "words = ['fig', 'apricot', 'plum', 'kiwi']\nprint(sorted(words, key=len))",
 'plum和kiwi长度一样，谁应该在前？请给出完整结果，并解释依据。', "['fig', 'plum', 'kiwi', 'apricot']",
 '按长度升序排列；排序稳定，长度相同的plum和kiwi保留原来的先后顺序。'),
('state', 'value = 2\nfor x in [11, 13, 17]:\n    value = value * 2 + x\nprint(value)',
 '我每一轮都从初值2开始算，这样对吗？请给出各轮的新值和最后结果。', '103',
 '下一轮使用上一轮更新后的值，三轮依次为15、43、103，不能每轮重置为2。')]

def main():
    if (DATA/'independent_test_manifest.json').exists():
        raise SystemExit('Natural-expression tests are frozen; existing test release must not be overwritten.')
    exports=[]; metadata=[]; errors=[]
    for i,(family,code,request,gold,explanation) in enumerate(CASES,1):
        out=io.StringIO()
        with redirect_stdout(out): exec(compile(ast.parse(code),'<authored-probe>','exec'),{})
        actual=out.getvalue().strip()
        if actual!=gold: errors.append({'id':i,'authored':gold,'actual':actual})
        prompt=request+'\n```python\n'+code+'\n```'
        chosen='本题输出：'+gold+ ('\n'+explanation if explanation else '')
        exports.append({'messages':[{'role':'user','content':prompt},{'role':'assistant','content':chosen}]})
        metadata.append({'candidate_id':f'V2-NATURAL-{i:02d}','family':'authored_'+family,'topic':family,
                         'level':'混合','task':'predict' if not explanation else 'explain',
                         'split':'independent_test','export_line':i,'prompt':prompt,'chosen':chosen,
                         'code':code,'answer':gold,'author':'Codex; individually authored without curriculum calls',
                         'intended_use':'final evaluation only; not training or epoch selection'})
    if errors: raise ValueError(errors)
    for name,records in [('independent_test.jsonl',exports),('independent_test_metadata.jsonl',metadata)]:
        (DATA/name).write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in records),encoding='utf8')
    manifest={'count':24,'reference_execution_checks':24,'errors':[],
              'frozen_before_observing_model_outputs':True,
              'author_scope':'Same Codex author as project; different construction from generator, not an external or human evaluation.',
              'test_role':'Natural-expression probes. Familiar concepts are allowed; this is not family-level OOD testing.',
              'sha256':{name:hashlib.sha256((DATA/name).read_bytes()).hexdigest() for name in ('independent_test.jsonl','independent_test_metadata.jsonl')}}
    (DATA/'independent_test_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(manifest,ensure_ascii=False))

if __name__=='__main__': main()
