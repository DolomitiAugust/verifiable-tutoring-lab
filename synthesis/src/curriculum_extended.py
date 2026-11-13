"""Additional small programs; expected outputs are computed algebraically before execution.

Three new families are development-only. The three v1 test families stay frozen.
"""
import random
from collections import Counter
from curriculum_base import FAMILIES as OLD_FAMILIES, make_case as old_case, _lesson

TEST_FAMILIES = ('snapshot', 'split_join', 'zip_pairs')
DEV_FAMILIES = ('boolean_logic', 'string_count', 'while_counter')
NEW_LEVELS = {'boolean_logic':'基础', 'string_count':'进阶', 'while_counter':'挑战',
              'list_comprehension':'进阶', 'set_unique':'基础', 'sorted_key':'进阶',
              'any_all':'基础', 'function_return':'进阶', 'accumulate':'挑战',
              'dict_comprehension':'挑战', 'unpack':'基础', 'loop_else':'挑战'}
FAMILIES = {**OLD_FAMILIES, **NEW_LEVELS}
TRAIN_FAMILIES = tuple(sorted(set(FAMILIES)-set(TEST_FAMILIES)-set(DEV_FAMILIES)))


def make_case(family, r):
    if family in OLD_FAMILIES:
        return old_case(family,r)
    n, k = r.randint(3,12), r.randint(2,6)
    vals = r.sample(range(1,30),r.randint(3,7))
    if family == 'boolean_logic':
        x,y=r.choice((True,False)),r.choice((True,False)); op=r.choice(('and','or'))
        ans=(x and y) if op=='and' else (x or y)
        other='or' if op=='and' else 'and'; trans=(x or y) if other=='or' else (x and y)
        code=f'a = {x}\nb = {y}\nprint(a {op} b)'
        data=_lesson('布尔运算','区分and和or',code,ans,not ans,
            f'a 为 {x}，b 为 {y}；and 要求两者都为真，or 只要求至少一个为真，所以这里得到 {ans}。',
            '混淆了与和或的条件。','当前运算要求两个条件同时成立吗？',
            f'把 {op} 换成 {other}。',code.replace(f'a {op} b',f'a {other} b'),trans)
    elif family=='string_count':
        text=''.join(r.choices('ab',k=n)); needle=r.choice(('a','b','ab','aa'))
        # Independent non-overlapping scan, not str.count.
        pos=0; ans=0
        while pos<=len(text)-len(needle):
            hit=text[pos:pos+len(needle)]==needle
            ans+=int(hit); pos+=len(needle) if hit else 1
        code=f'text = {text!r}\nprint(text.count({needle!r}))'
        trans=ans+1
        data=_lesson('子串计数','理解不重叠计数',code,ans,ans+1,
            f'从左向右匹配 {needle!r}，每次匹配后跳过整个片段，得到 {ans} 次不重叠匹配。',
            '把不存在或重复覆盖的匹配也计入了。','一次匹配成功后应从哪里继续？',
            '在字符串末尾隔开一个#再追加同样片段。',
            f'text = {text!r} + "#" + {needle!r}\nprint(text.count({needle!r}))',trans)
    elif family=='while_counter':
        start=r.randint(1,5); stop=start+k*n
        ans=sum(start+k*i for i in range(n))
        code=f'i = {start}\ntotal = 0\nwhile i < {stop}:\n    total += i\n    i += {k}\nprint(total)'
        data=_lesson('while状态更新','追踪终止条件',code,ans,ans+stop,
            f'i 依次取 {[start+k*i for i in range(n)]}，到 {stop} 时条件为假，累计值为 {ans}。',
            '把等于终点的那一轮也执行了。','检查是在本轮累加之前还是之后？',
            '把小于改为小于等于。',code.replace(' < ',' <= '),ans+stop)
    elif family=='list_comprehension':
        ans=[v*k for v in vals if v%2==0]; wrong=[v*k for v in vals]
        if ans==wrong: vals.append(1); ans=[v*k for v in vals if v%2==0]; wrong=[v*k for v in vals]
        code=f'values = {vals}\nprint([x * {k} for x in values if x % 2 == 0])'
        trans=[v*k for v in vals if v%2!=0]
        data=_lesson('列表推导式','先筛选再变换',code,ans,wrong,
            f'只保留偶数 {[v for v in vals if v%2==0]}，逐个乘 {k}，结果是 {ans}。',
            '忽略了末尾的筛选条件。','if 条件控制哪些元素进入结果？',
            '把偶数条件改成奇数条件。',code.replace('== 0','!= 0'),trans)
    elif family=='set_unique':
        vals=vals+[vals[0],vals[-1]]; ans=len(dict.fromkeys(vals)); wrong=len(vals)
        code=f'values = {vals}\nprint(len(set(values)))'
        data=_lesson('集合去重','区别元素总数和不同值个数',code,ans,wrong,
            f'不同的值是 {list(dict.fromkeys(vals))}，重复值只保留一次，因此有 {ans} 个。',
            '按列表总长度计算，没有去掉重复值。','同一个值在集合里会保留几份？',
            '改成直接求原列表长度。',code.replace('len(set(values))','len(values)'),wrong)
    elif family=='sorted_key':
        words=r.sample(['fig','pear','apple','banana','kiwi','watermelon','plum','cherry'],4)
        reverse=r.choice((True,False)); ans=sorted(words,key=len,reverse=reverse)
        wrong=sorted(words,reverse=reverse)
        if ans==wrong: return make_case(family,r)
        code=f'words = {words!r}\nprint(sorted(words, key=len, reverse={reverse}))'
        data=_lesson('排序键','按指定键稳定排序',code,ans,wrong,
            f'比较的是字符数 {[len(x) for x in words]}，reverse={reverse}；长度相同时保留原顺序，得到 {ans}。',
            '按字母顺序排列，忽略了key=len。','排序比较的是单词本身还是长度？',
            '只把reverse取反。',code.replace(f'reverse={reverse}',f'reverse={not reverse}'),sorted(words,key=len,reverse=not reverse))
    elif family=='any_all':
        flags=[True,False]+r.choices((True,False),k=r.randint(1,4)); r.shuffle(flags)
        op=r.choice(('any','all')); ans=op=='any'; other='all' if op=='any' else 'any'
        code=f'flags = {flags}\nprint({op}(flags))'
        data=_lesson('any与all','区分至少一个与全部',code,ans,not ans,
            f'列表既含True也含False。any只需一个True，all要求全部True，所以 {op} 返回 {ans}。',
            '把至少一个和全部的要求弄反了。','一个False是否足以决定当前函数的结果？',
            f'改用{other}。',code.replace(f'{op}(flags)',f'{other}(flags)'),not ans)
    elif family=='function_return':
        bias=r.randint(1,9); ans=n*k+bias; wrong=n*k
        code=f'def scale(x):\n    return x * {k} + {bias}\nprint(scale({n}))'
        data=_lesson('函数返回值','代入参数并计算返回表达式',code,ans,wrong,
            f'实参 {n} 绑定到x，返回 {n} × {k} + {bias} = {ans}。',
            '乘法后漏加了偏移量。','return后面是完整表达式还是只有乘法？',
            '实参增加1。',code.replace(f'scale({n})',f'scale({n+1})'),ans+k)
    elif family=='accumulate':
        initial=r.randint(1,5); ans=initial
        for v in vals: ans=ans*k+v
        wrong=initial+sum(v*k for v in vals)
        code=f'value = {initial}\nfor x in {vals}:\n    value = value * {k} + x\nprint(value)'
        states=[]; s=initial
        for v in vals: s=s*k+v; states.append(s)
        data=_lesson('累积状态','使用更新后的值进入下一轮',code,ans,wrong,
            f'每轮都用上一轮的新值；各轮结果依次为 {states}，最后是 {ans}。',
            '把各轮当作互不依赖的加法。','下一轮value是初值还是上一轮结果？',
            '初值增加1。',code.replace(f'value = {initial}',f'value = {initial+1}',1),ans+k**len(vals))
    elif family=='dict_comprehension':
        modulus=r.randint(2,4); ans={}
        for v in vals: ans[v%modulus]=v
        wrong={v:v%modulus for v in vals}
        code=f'values = {vals}\nprint({{x % {modulus}: x for x in values}})'
        reverse={}
        for v in reversed(vals): reverse[v%modulus]=v
        data=_lesson('字典推导式','识别键值与重复键覆盖',code,ans,wrong,
            f'冒号左侧是余数键，右侧是原值；相同键后写覆盖先写，最终字典为 {ans}。',
            '交换了冒号两侧的键和值。','相同余数的两个元素会占两个键吗？',
            '反向遍历原列表。',code.replace('in values','in reversed(values)'),reverse)
    elif family=='unpack':
        vals=r.sample(range(1,30),r.randint(4,8)); ans=sum(vals[1:-1]); wrong=sum(vals)
        code=f'first, *middle, last = {vals}\nprint(sum(middle))'
        data=_lesson('星号解包','区分首尾与中间部分',code,ans,wrong,
            f'首项 {vals[0]} 和末项 {vals[-1]} 单独取走，middle是 {vals[1:-1]}，其和为 {ans}。',
            '把首尾两项也算进middle。','星号变量是否还包括已分配的首尾？',
            '改为打印first与last之和。',code.replace('sum(middle)','first + last'),vals[0]+vals[-1])
    elif family=='loop_else':
        needle=r.choice(vals+[50]); found=needle in vals
        code=f'found = False\nfor x in {vals}:\n    if x == {needle}:\n        found = True\n        break\nelse:\n    found = False\nprint(found)'
        data=_lesson('for-else','识别break与else的关系',code,found,not found,
            f'{needle} '+('在列表中，匹配时执行break，跳过else，所以found为True。' if found else '不在列表中，循环正常结束并执行else，所以found为False。'),
            '把循环else当成每一轮if的else。','循环是正常耗尽还是被break终止？',
            '把要找的值改为列表第一项。',code.replace(f'x == {needle}',f'x == {vals[0]}'),True)
    else: raise KeyError(family)
    return {'family':family,'level':NEW_LEVELS[family],**data}
