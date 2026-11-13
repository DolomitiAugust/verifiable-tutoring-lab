"""Render Chinese tutoring examples from cases with independently computed outputs."""


_LEARNERS = {
    "spreadsheet": (
        (
            "我是经常用电子表格做作业的大一学生，刚开始学 Python。",
            "做实验时我习惯把每一步的数记在表格里，Python 对我还是新东西。",
            "我会用表格公式算结果，读程序时却常常只看最后一行。",
        ),
        (
            "你可以借用核对表格的办法，把“读到哪里”和“此时的值”并排记，只填真正发生变化的地方。",
            "在纸上留三列：关键语句、执行前、执行后。在结果开始与预测不同的那一行停下来。",
            "把最后的输出看成表格的末行，然后往上查第一处不一致的记录，比只改末行的答案更有用。",
        ),
    ),
    "blocks": (
        (
            "我在 Scratch 里拼过积木，但刚开始读 Python 的文字代码。",
            "我能看懂积木程序的执行顺序，换成 Python 后容易看漏动作。",
            "我是刚从图形化编程转学 Python 的学生，习惯跟着每个动作往下走。",
        ),
        (
            "像给积木程序单步播放一样，先圈出决定结果的动作。完成这个动作后，再读下一步。",
            "用手指沿着实际执行的顺序移动。遇到你认为会影响结果的动作就停一下，说出它究竟改变了什么。",
            "把每个关键动作画成一块积木，给真正执行过的动作打钩。写出最后的结果前，再核对打钩的顺序。",
        ),
    ),
    "math": (
        (
            "我习惯先在纸上做数学推导，刚学 Python，有时会把两者的写法混起来。",
            "我数学基础还行，读代码时常凭公式直觉猜输出。",
            "我是第一次上程序设计课的数学系学生，喜欢从一个具体例子总结规则。",
        ),
        (
            "沿用你做推导的习惯，写出一行中间式，再在旁边注明用了哪条 Python 规则。",
            "先用这组具体输入手算一遍，再想这一步在什么条件下仍成立。不要直接把一次观察写成通则。",
            "把你的预测当作一个待检验的假设，找到程序里第一个能区分两种推导的地方。",
        ),
    ),
    "self_taught": (
        (
            "我自学过一点 Python，平时靠反复运行改代码，但不太会解释原因。",
            "我做过几道 Python 练习，习惯看运行结果再改，想练习先预测。",
            "我会在代码里加 print 调试，可是一拿走电脑就不确定每步发生了什么。",
        ),
        (
            "这次先不运行，把你预计看到的值写在关键语句旁。最后运行一次，只查第一处不同的记录。",
            "用你熟悉的调试办法，不过先在纸上设一个断点，说出预期值，再用一次运行核对。",
            "先预测，再运行，一次只改变一个条件。两次结果有差别时，回到造成差别的那一步。",
        ),
    ),
}

PROFILES = tuple(_LEARNERS)
TASKS = ("explain", "diagnose", "quiz", "coach")
PREF_REASONS = ("factual_error", "incomplete", "off_target")

_EVIDENCE = {
    "explain": ("none", "incorrect", "correct", "none", "incorrect", "none", "incorrect", "none"),
    "diagnose": ("incorrect", "correct", "incorrect", "incorrect", "correct", "incorrect"),
    "quiz": ("none", "none", "incorrect", "none", "correct", "none", "none", "incorrect"),
    "coach": ("correct", "incorrect", "incorrect", "correct", "incorrect", "incorrect"),
}


def evidence_kind(task, variant):
    """Describe the information actually supplied by the learner in this instruction."""
    choices = _EVIDENCE[task]
    return choices[variant % len(choices)]


_REQUESTS = {
    "explain": (
        "请讲讲程序真正打印什么、这个结果怎样得到，再出一道相近的题，附参考答案。",
        "请用我能跟上的方式解释运行过程，指出我可能想错的地方。最后给一个带答案的变式。",
        "请带我推一遍关键步骤，让我看懂结果。然后换一个条件考我，参考答案另列。",
    ),
    "diagnose": (
        "请对照我的预测找出可能出错的那一步，说明依据，再给一题带答案的复测。",
        "我的预测可能错在哪里？请给出真实输出，解释为什么，再让我用一个有参考答案的变式检验。",
        "请先指出我的判断在哪条规则上可能走偏，再对照真实结果讲解，并附一道带答案的检验题。",
    ),
    "quiz": (
        "请把这段程序编成一份两小题的测验：先预测原题，再做变式。把题面和核对用的答案分开。",
        "请出两道预测输出题，以原程序为第一题，只改一个条件作为第二题。参考答案放到最后，讲解第一题的易错处。",
        "请按我的程度把这段代码做成原题和变式两小题，便于我先独立作答；另给教师用答案和原题讲评。",
    ),
    "coach": (
        "请根据这一次预测给我学习反馈，找出要复习的地方，安排下一道练习并附核对答案。",
        "我想知道这次结果反映出哪个知识点值得复习。请说明真实输出，告诉我下一步怎么练，并给出有答案的练习。",
        "请就这一次作答给我具体反馈，再安排一道能检验我有没有理解的复测，附参考答案。",
    ),
}

_CHECK_REQUESTS = {
    "explain": (
        "请帮我核对预测，并示范关键步骤。最后再给一道带答案的变式。",
        "我想知道预估的结果有没有依据，请讲解，再留一道有参考答案的练习。",
    ),
    "diagnose": (
        "请核对我的预测。如果值对了，请指出支撑它的关键证据；如果不对，说明可能错在哪里。再给一道有答案的复测。",
        "我写了结果但还没想好理由。请对照代码检查我的判断，再给一道变式和参考答案。",
    ),
    "coach": (
        "我只有这一次预测可以参考。请评价它，说明接下来该验证什么，再给一道有答案的复测。",
        "请看我的预估，给我针对这次表现的反馈和下一步练习，最后附上练习答案。",
    ),
}

_QUIZ_STEMS = {
    "基础": "先写出这段程序打印的内容，再指出决定结果的一步。",
    "进阶": "写出运行结果，并挑出一个中间状态，说明它怎样影响最后的输出。",
    "挑战": "不运行程序，预测输出。列出至少一个中间值，说明哪一步决定了最终结果。",
}


def _pick(options, variant):
    return options[variant % len(options)]


def _prompt(case, profile, task, variant):
    intro = _pick(_LEARNERS[profile][0], variant)
    kind = evidence_kind(task, variant)
    request = _pick(_REQUESTS[task], variant)
    if kind == "correct" and task in _CHECK_REQUESTS:
        request = _pick(_CHECK_REQUESTS[task], variant)
    elif kind == "none" and task == "explain":
        request = "请带我推一遍程序，说明真正的输出和关键依据，再给一道变式和参考答案。"
    code = f"```python\n{case['code']}\n```"
    goal = f"{case['topic']}（{case['level']}）\n学习目标：{case['objective']}"
    observed = case["answer"] if kind == "correct" else case["wrong"]
    guess = f"我预测程序会打印：{observed}" if kind != "none" else ""
    layouts = (
        _parts(f"{intro}\n这次练习的是{goal}", code, guess, request),
        _parts(f"{intro}\n{request}", f"练习主题：{goal}", code, guess),
        _parts(f"{intro}\n主题：{goal}", guess, f"我读到的代码是：\n{code}", request),
    )
    return _pick(layouts, variant)


def _output(value):
    return f"本题输出：{value}"


def _practice(case):
    return f"换一道题试试：{case['transfer']}\n变式参考输出：{case['transfer_answer']}"


def _diagnosis(case, kind):
    if kind == "incorrect":
        return f"从你给的预测「{case['wrong']}」看，可以先检查这个误解：{case['misconception']}"
    if kind == "correct":
        return f"你预测的「{case['answer']}」与这次输出一致。复核理由时，注意一个容易混淆的地方：{case['misconception']}"
    return f"本题常见的一种误读：{case['misconception']}"


def _quiz(case):
    return (
        f"第一题（{case['level']}）　{_QUIZ_STEMS[case['level']]}\n"
        f"```python\n{case['code']}\n```\n"
        f"第二题　{case['transfer']}"
    )


def _parts(*parts):
    return "\n\n".join(part for part in parts if part)


_SELF_CHECK = {
    "spreadsheet": "准备一张三列表格。第一列抄关键动作，第二列写动作之前有关的值，第三列写动作之后的值。每次只核对发生变化的那一格，用一条线连到下一行。程序读完，再拿末行和自己的预测逐字比对。",
    "blocks": "先用几张小纸条代替积木，只写本题真正会影响输出的动作，按执行顺序排在桌上。每移动一张，就在纸条下方写一个当前值。两张纸条之间若说不出为什么会变化，就回到那一行代码再看一次。",
    "math": "可以把草稿分成两栏：左侧只列题目给出的值和运算，右侧写每一步所用的 Python 规则。先算出具体结果，再检查删掉哪一步会改变结果。这样也能看出刚才的推算是用了规则，还是恰巧碰上同一个值。",
    "self_taught": "先遮住答案，选一个会影响输出的位置，在草纸上写下预测值，再自己运行原程序。两者一致就说明这一次的预测经得起核验；不同就找最早出现差异的语句，只改一条推算。这比同时改几处代码更容易看清原因。",
}


def _study_style(text, profile, variant, extended):
    """Use the same stock of short and detailed explanations on both sides."""
    short = _pick(_LEARNERS[profile][1], variant)
    long = _SELF_CHECK[profile]
    return text.replace(short, long, 1) if extended else text.replace(long, short, 1)


def render(case, profile, task, variant: int):
    """Return an instruction and its reference response. Both depend on the task."""
    prompt = _prompt(case, profile, task, variant)
    output, trace = _output(case["answer"]), case["reasoning"]
    kind = evidence_kind(task, variant)
    diagnosis, practice = _diagnosis(case, kind), _practice(case)
    scaffold = _pick(_LEARNERS[profile][1], variant)
    hint = case["hint"]

    if task == "explain":
        if variant % 2:
            body = _parts(scaffold, trace, output, f"{diagnosis}\n可以这样自查：{hint}", practice)
        else:
            body = _parts(output, trace, diagnosis, f"{scaffold}\n{hint}", practice)
    elif task == "diagnose":
        body = _parts(diagnosis, f"回到代码核对：{trace}", output, f"{scaffold}\n{hint}", practice)
    elif task == "quiz":
        body = _parts(
            _quiz(case),
            f"教师核对用\n{output}\n{trace}\n变式参考输出：{case['transfer_answer']}",
            f"第一题讲评：{diagnosis}\n{scaffold} {hint}",
        )
    elif task == "coach":
        body = _parts(
            f"这次先复习「{case['topic']}」。{diagnosis}",
            f"{trace}\n{output}",
            f"练习方法：{scaffold}\n做题时先问自己：{hint}",
            practice,
        )
    else:
        raise ValueError(f"Unknown task: {task!r}")
    return prompt, _study_style(body, profile, variant, variant % 4 in (0, 2))


def _factual(case, profile, task, variant):
    """Present the generator's alternative misconception as the rule for this case."""
    wrong_output = _output(case["wrong"])
    scaffold = _pick(_LEARNERS[profile][1], variant)
    kind = evidence_kind(task, variant)
    verdict = {
        "none": "先算原题，变式单独核对。",
        "incorrect": f"你给的「{case['wrong']}」可以作为这道题的参考结果。",
        "correct": f"你预测的是「{case['answer']}」。按下面的步骤检查，会得到另一个结果。",
    }[kind]
    rule = _alternative_rule(case)
    rationale = f"{verdict}{rule}按这种读法，原题最后会显示「{case['wrong']}」。"
    if task == "quiz":
        return _parts(
            _quiz(case),
            f"教师核对用\n{wrong_output}\n{rationale}\n变式参考输出：{case['transfer_answer']}",
            f"讲评时让学生说明预测依据。{scaffold}",
        )
    if task == "diagnose":
        return _parts(
            rationale,
            wrong_output, f"自查方法：{scaffold}", _practice(case),
        )
    if task == "coach":
        return _parts(
            f"这次可以从「{case['topic']}」着手复核。{rationale}",
            wrong_output, f"下一次可以这样练：{scaffold}", _practice(case),
        )
    return _parts(wrong_output, rationale, scaffold, _practice(case))


def _alternative_rule(case):
    code = case["code"]
    family = case["family"]
    alternatives = {
        "precedence": ("先做括号右边的乘法，再与左边的数合并。" if "print((" in code
                       else "从左往右先把前两个数合并，再乘最后一个数。"),
        "division": ("双斜杠在这里保留分组后的余量。" if "//" in code
                     else "百分号在这里报告能装满多少整组。"),
        "snapshot": "后一次修改会一起更新之前保存的整数变量，所以在两个名字处都使用新值。",
        "slice": "起点和终点这两个位置的字符都要取到，再按步长连成字符串。",
        "index": "正下标可以从第一项记作 1 开始数；负下标则把最末项记作 0，再向前数。",
        "boundary": ("即使两边相等，眼前的比较条件也不成立，所以取 else 分支。"
                     if ">=" in code or "<=" in code else
                     "边界值相等时可以进入 if 分支，所以取这个分支的赋值。"),
        "range_sum": "从起点开始按步长往前，恰好到终点时还要再累加一次。",
        "filter": ("通过筛选的元素应按各自的原值累加。" if "total += 1" in code
                   else "每当一个元素通过筛选，就只给计数加一。"),
        "split_join": "应把 split 得到的全部片段交给 join，输出时所有片段都会出现。",
        "alias": ("切片后的 b 与 a 仍引用同一张列表，因此 b 的修改也会出现在 a 里。"
                  if "b = a[:]" in code else
                  "给 b 赋值会保存一张独立的列表，之后对 b 的修改不会反映到 a。"),
        "default_arg": ("调用传了第二个值，但计算返回值时仍使用定义处的默认值。"
                        if "," in code.splitlines()[-1] else
                        "调用没有提供第二个参数，就按零来计算对应的这一项。"),
        "branch_order": "两个条件都成立，因此后面的 elif 赋值会覆盖前一个分支的结果。",
        "nested_pairs": "内层每遍历到一个值就计入一对，最后应把两个列表的长度相乘。",
        "loop_control": ("break 会略过当前这个数，后面的数仍要继续累加。"
                         if "break" in code else
                         "continue 在触发点就停止整个循环，后面的数不用再计算。"),
        "recursion": "调用抵达停止条件时只负责结束递归，返回阶段只统计前面各层的贡献。",
        "enumerate": "enumerate 的位置编号从零递增，所以权重的第一项应该使用零。",
        "dictionary": "同名键多次出现时这里只保留它最后一次读到的 value。",
        "zip_pairs": "较长列表还剩一个数时，会自动给它配一个中性值，末项仍要计入总和。",
    }
    return alternatives.get(family, "沿着原先的求值顺序，可以保留这个预测。")


def _incomplete(case, profile, task, variant, chosen):
    """Give the true output, while leaving out a deliverable specific to the task."""
    output = _output(case["answer"])
    scaffold = _pick(_LEARNERS[profile][1], variant)
    if task in ("explain", "coach"):  # Accurate lesson or feedback, without the requested transfer.
        return chosen.replace(f"\n\n{_practice(case)}", "", 1)
    if task == "diagnose":  # Correct conclusion and follow-up, without diagnosis / supporting trace.
        return _parts(output, scaffold, _practice(case))
    if task == "quiz":  # Sets both questions, but the key covers only the first.
        return chosen.replace(f"\n变式参考输出：{case['transfer_answer']}", "", 1)
    raise ValueError(f"Unknown task: {task!r}")


_SIDE_LESSONS = (
    (
        "怎样控制多个值之间的分隔符", "print('A', 'B', sep='-')",
        "A-B", "sep 决定同一次 print 调用里多个对象之间放什么字符。这里有两个对象，因此中间只出现一次连字符。",
        "保持两个字符串不变，只把 sep 改成斜杠。先预测，再运行核对；结果会是 A/B。"
    ),
    (
        "两次输出怎样排在同一行", "print('A', end='')\nprint('B')",
        "AB", "第一次调用把结尾设置为空字符串，所以它不会先换行。第二次调用从同一行接着显示 B。",
        "把第一行的 end 改成一个空格，第二行保持原样。输出会是 A B，两者在同一行，中间留一个空格。"
    ),
    (
        "怎样区分显示的文字和注释", "print('A#B')  # 记录标签",
        "A#B", "引号里的井号属于字符串，会原样显示。右括号之后的井号才开始一段注释，因此后面的文字不进入输出。",
        "把标签改成 A#C，只改引号内的 B。输出变为 A#C；把右侧注释改为另一句话，输出仍然一样。"
    ),
)


def _off_target(case, profile, task, variant):
    """Provide a valid nearby Python lesson that misses the requested learning objective."""
    options = _SIDE_LESSONS[1:] if case["family"] == "split_join" else _SIDE_LESSONS
    subject, code, answer, rule, exercise = _pick(options, variant)
    output = _output(case["answer"])
    scaffold = _pick(_LEARNERS[profile][1], variant)
    snippet = f"```python\n{code}\n```"
    activity = f"把这个例子的结果记在学习记录里。\n{snippet}\n参考输出是「{answer}」。{rule}"
    if task == "quiz":
        return _parts(
            f"热身题　预测题干中原程序的输出。\n扩展题　练习「{subject}」，预测下面的显示内容，并说明理由。\n{snippet}",
            f"教师核对用\n{output}\n扩展题输出：{answer}\n{rule}",
            f"讲评时可再试一个变化：{exercise}\n{scaffold}",
        )
    if task == "diagnose":
        lead = f"已把原题的结果列在下面。我们接着从屏幕上怎样显示字符入手，把「{subject}」单独练一次。"
    elif task == "coach":
        lead = f"下一步练习「{subject}」。先把原程序的结果记好，再看下面这个专门观察显示形式的小实验。"
    else:
        lead = f"可以从「{subject}」入手理解屏幕上的文字。先记下原程序的输出，再观察下面这个短例子。"
    return _parts(lead, output, activity, f"动手验证：{exercise}", scaffold)


def reject(case, profile, task, chosen, reason, variant: int):
    """Return a candidate with the requested documented failure for this instruction."""
    if reason == "factual_error":
        result = _factual(case, profile, task, variant)
    elif reason == "incomplete":
        result = _incomplete(case, profile, task, variant, chosen)
    elif reason == "off_target":
        result = _off_target(case, profile, task, variant)
    else:
        raise ValueError(f"Unknown preference reason: {reason!r}")
    return _study_style(result, profile, variant, variant % 4 in (1, 2))
