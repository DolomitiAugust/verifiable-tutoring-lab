"""可复现的大学 Python 入门程序题。数值答案独立算出，不通过执行题面取得。"""

from bisect import bisect_left, bisect_right
from math import lcm
import random


LEVELS = ("基础", "进阶", "挑战")


def _lesson(topic, objective, code, answer, wrong, reasoning, misconception,
            hint, transfer, transfer_code, transfer_answer):
    return {
        "topic": topic, "objective": objective, "code": code,
        "answer": str(answer), "wrong": str(wrong), "reasoning": reasoning,
        "misconception": misconception, "hint": hint, "transfer": transfer,
        "transfer_code": transfer_code, "transfer_answer": str(transfer_answer),
    }


def _precedence(r):
    a, b, scale = r.randint(3, 40), r.randint(2, 19), r.randint(2, 9)
    sign = r.choice(("+", "-"))
    signed_b = b if sign == "+" else -b
    flat, grouped = f"{a} {sign} {b} * {scale}", f"({a} {sign} {b}) * {scale}"
    direct, grouped_value = a + signed_b * scale, (a + signed_b) * scale
    if r.choice((False, True)):
        expr, other, ans, wrong = grouped, flat, grouped_value, direct
        explain = f"括号内 {a} {sign} {b} = {a + signed_b}，再乘 {scale}，得到 {ans}。"
        mistake = f"忽略了括号，把 {b} 与 {scale} 先相乘，因此会算成 {wrong}。"
    else:
        expr, other, ans, wrong = flat, grouped, direct, grouped_value
        explain = f"先算 {b} × {scale} = {b * scale}，再算 {a} {sign} {b * scale} = {ans}。"
        mistake = f"从左往右先算 {a} {sign} {b}，相当于给前两项多加了一对括号。"
    return _lesson("表达式的计算顺序", "根据括号和乘法优先级求值", f"print({expr})",
                   ans, wrong, explain, mistake, f"看 {expr}，哪一步应当先算？",
                   f"把表达式换成 {other}，会打印什么？", f"print({other})", wrong)


def _division(r):
    size, q = r.randint(3, 16), r.randint(2, 14)
    rem = r.choice([x for x in range(1, size) if x != q])
    n = q * size + rem
    amount, pack = r.choice((("books", "per_box"), ("students", "per_team"),
                             ("pages", "per_day"), ("tokens", "per_batch")))
    op = r.choice(("//", "%"))
    ans, wrong = (q, rem) if op == "//" else (rem, q)
    code = f"{amount} = {n}\n{pack} = {size}\nprint({amount} {op} {pack})"
    extra = size - rem
    return _lesson("整除与取余", "区分完整组数和剩余数量", code, ans, wrong,
                   f"{n} = {q} × {size} + {rem}。// 取完整组数，% 取余数，程序用 {op}，所以输出 {ans}。",
                   f"把整除和取余的作用弄反了。{wrong} 是另一种运算的结果。",
                   f"把 {n} 个物品每 {size} 个分一组，当前符号是取组数还是剩余数？",
                   f"若把 {amount} 增加 {extra}，正好凑满下一组，原程序会打印什么？",
                   code.replace(f"{amount} = {n}", f"{amount} = {n + extra}", 1),
                   q + 1 if op == "//" else 0)


def _snapshot(r):
    name, old = r.choice((("score", "previous"), ("stock", "saved"), ("seats", "initial")))
    n, change = r.randint(7, 50), r.randint(2, 8)
    op = r.choice(("+", "-", "*"))
    after = {"+": n + change, "-": n - change, "*": n * change}[op]
    solo = r.choice((False, True))
    out = old if solo else f"{old} + {name}"
    assignment, update = f"{old} = {name}", f"{name} {op}= {change}"
    prefix = f"{name} = {n}\n"
    code = prefix + f"{assignment}\n{update}\nprint({out})"
    ans, wrong = (n, after) if solo else (n + after, after * 2)
    transfer_code = prefix + f"{update}\n{assignment}\nprint({out})"
    return _lesson("赋值与更新", "追踪更新前后两个整数变量的值", code, ans, wrong,
                   f"{old} 先保存了 {n}。后来 {name} 变为 {n} {op} {change} = {after}，"
                   f"{old} 仍为 {n}，所以 {out} 打印 {ans}。",
                   f"把先前保存的 {old} 也当作 {after}，会得到 {wrong}。",
                   f"{old} = {name} 发生在更新之前还是之后？此后哪一个变量被重新赋值？",
                   f"若把“{assignment}”移到“{update}”之后，仍打印 {out}，会得到什么？",
                   transfer_code, wrong)


_WORDS = ("data", "type", "page", "code", "user", "note", "time", "quiz",
          "unit", "task", "flow", "loop", "file", "path", "read", "work")


def _slice(r):
    text = "".join(r.sample(_WORDS, 2))
    stride = r.choice((1, 2))
    start = r.randint(0, len(text) - stride - 1)
    stop = start + stride * r.randint(1, (len(text) - 1 - start) // stride)
    selector = f"{start}:{stop}" + ("" if stride == 1 else f":{stride}")
    other = f"{start}:{stop + 1}" + ("" if stride == 1 else f":{stride}")
    positions = list(range(start, stop, stride))
    ans = "".join(text[i] for i in positions)
    wrong = ans + text[stop]
    code = f"word = {text!r}\nprint(word[{selector}])"
    return _lesson("字符串切片", "识别切片的起点、终点和步长", code, ans, wrong,
                   f"下标从 0 开始。取下标 {positions} 的字符，遇到终点 {stop} 就停，拼起来是 {ans}。",
                   f"把终点 {stop} 也取了，就会多一个字符 {text[stop]}。",
                   f"从下标 {start} 起每次前进 {stride}，哪些下标严格小于 {stop}？",
                   f"把切片改为 word[{other}]，会打印哪串字符？",
                   code.replace(f"[{selector}]", f"[{other}]", 1), wrong)


def _index(r):
    values = r.sample(range(2, 80), r.randint(4, 8))
    idx = r.randint(1, len(values) - 2)
    if r.choice((False, True)):
        idx = -idx
    offset = idx if idx > 0 else len(values) + idx
    ans, wrong = values[offset], values[offset - 1]
    code = f"marks = {values}\nprint(marks[{idx}])"
    locate = f"正下标 {idx} 表示第 {idx + 1} 项" if idx > 0 else (
        f"负下标 {idx} 表示倒数第 {-idx} 项，换成正下标是 {len(values)} + ({idx}) = {offset}")
    mistake = (f"从 1 开始数会错取第 {offset} 项。" if idx > 0 else
               f"把最后一项当作下标 0 向前数，会多退一格，取到 {wrong}。")
    adjacent = offset + 1 if offset < len(values) - 1 else offset - 1
    return _lesson("列表下标", "用正下标和负下标定位一项", code, ans, wrong,
                   f"{locate}。这项的值为 {ans}，所以 print 输出 {ans}。",
                   mistake, "列表第一项的下标是多少？最后一项用什么负下标表示？",
                   f"把 {idx} 改为正下标 {adjacent}，此时会打印什么？",
                   code.replace(f"marks[{idx}]", f"marks[{adjacent}]", 1), values[adjacent])


def _boundary(r):
    cutoff = r.randint(30, 95)
    op = r.choice((">=", ">", "<=", "<"))
    accepted = op in (">=", "<=")
    reward, fallback = r.sample(range(1, 31), 2)
    name = r.choice(("score", "attendance", "progress"))
    code = (f"{name} = {cutoff}\nif {name} {op} {cutoff}:\n    points = {reward}\n"
            f"else:\n    points = {fallback}\nprint(points)")
    ans, wrong = (reward, fallback) if accepted else (fallback, reward)
    # Move the value just across the boundary so the alternative branch runs.
    delta = -1 if op in (">=", "<") else 1
    return _lesson("条件的边界", "辨别比较符号是否包含相等情况", code, ans, wrong,
                   f"比较的是 {cutoff} {op} {cutoff}，它是 {accepted}。程序走"
                   f" {'if' if accepted else 'else'} 分支，points 取 {ans}。",
                   f"在两个数相等时把条件判断反了，就会走另一个分支，输出 {wrong}。",
                   f"两个操作数都是 {cutoff}，符号 {op} 包含等号吗？",
                   f"若把第一行的 {name} 改为 {cutoff + delta}，其余不变，会打印什么？",
                   code.replace(f"{name} = {cutoff}", f"{name} = {cutoff + delta}", 1), wrong)


def _range_sum(r):
    start = r.randint(2, 30)
    step = r.choice((-5, -4, -3, -2, -1, 1, 2, 3, 4, 5))
    count, initial = r.randint(3, 7), r.randint(0, 10)
    stop = start + step * count
    if stop == 0:
        start, stop = start + 1, stop + 1
    nums = [start + step * i for i in range(count)]
    expr = f"range({start}, {stop}, {step})"
    code = f"total = {initial}\nfor n in {expr}:\n    total += n\nprint(total)"
    ans = initial + sum(nums)
    next_expr = f"range({start}, {stop}, {step * 2})"
    change = f"步长从 {step} 改为 {step * 2}"
    next_ans = initial + sum(nums[::2])
    if next_ans == ans:
        next_expr = f"range({start}, {stop + step}, {step})"
        change = f"终点从 {stop} 改为 {stop + step}"
        next_ans = ans + stop
    return _lesson("range 的终点", "追踪含正步长或负步长的循环", code, ans,
                   ans + stop, f"{expr} 依次取 {nums}。加上初值 {initial}，总和为 {ans}。",
                   f"把终点 {stop} 也加进去了。range 到达终点前就停止。",
                   f"从 {start} 开始，每次加 {step}，最后一次还能取到 {stop} 吗？",
                   f"若把 {expr} 的{change}，最后打印什么？",
                   code.replace(expr, next_expr, 1), next_ans)


def _filter(r):
    mod = r.randint(2, 6)
    remainder = r.randrange(mod)
    keep = r.sample([x for x in range(6, 80) if x % mod == remainder], r.randint(2, 4))
    skip = r.sample([x for x in range(6, 80) if x % mod != remainder], r.randint(1, 3))
    vals = keep + skip
    r.shuffle(vals)
    counting = r.choice((False, True))
    expr = "1" if counting else "n"
    if r.choice((False, True)):
        loop = f"    if n % {mod} == {remainder}:\n        total += {expr}"
    else:
        loop = f"    if n % {mod} != {remainder}:\n        continue\n    total += {expr}"
    code = f"values = {vals}\ntotal = 0\nfor n in values:\n{loop}\nprint(total)"
    ans, wrong = (len(keep), sum(keep)) if counting else (sum(keep), len(keep))
    extra = max(keep) + mod
    kept_in_order = [x for x in vals if x in keep]
    return _lesson("筛选后计数或求和", "先判断余数，再辨别循环加的是 1 还是原值", code, ans, wrong,
                   f"除以 {mod} 余 {remainder} 的数依次是 {kept_in_order}。"
                   f"循环每次加 {expr}，最后得到 {ans}。",
                   f"满足条件的有 {len(keep)} 个，它们的和为 {sum(keep)}。混淆数量和总和会得到 {wrong}。",
                   f"哪些数除以 {mod} 余 {remainder}？这些轮次里 total 每次增加什么？",
                   f"在 values 的末尾加入 {extra}，其余不变，最后打印什么？",
                   code.replace(f"values = {vals}", f"values = {vals + [extra]}", 1),
                   ans + (1 if counting else extra))


def _split_join(r):
    words = r.sample(_WORDS, r.randint(4, 7))
    split_by, join_by = r.sample((":", "/", "-", "_", "+", "|"), 2)
    cut = r.choice(("1:", ":-1", "::2"))
    chosen = {"1:": words[1:], ":-1": words[:-1], "::2": words[::2]}[cut]
    ans, wrong = join_by.join(chosen), join_by.join(words)
    code = (f"line = {split_by.join(words)!r}\nparts = line.split({split_by!r})\n"
            f"picked = parts[{cut}]\nprint({join_by!r}.join(picked))")
    return _lesson("拆分与拼接字符串", "识别 split、切片和 join 各做了什么", code, ans, wrong,
                   f"先按 {split_by!r} 拆成 {words}，切片 [{cut}] 保留 {chosen}，"
                   f"再用 {join_by!r} 连接，得到 {ans}。",
                   f"跳过 picked 的切片会把所有片段都拼回去，从而得到 {wrong}。",
                   "先手写 parts 的内容。picked 在这些片段中保留了哪几个？",
                   "若把最后一行 join 的参数从 picked 改为 picked[::-1]，会打印哪串字符？",
                   code.replace(".join(picked)", ".join(picked[::-1])", 1), join_by.join(reversed(chosen)))


def _alias(r):
    vals = r.sample(range(2, 60), r.randint(3, 6))
    extra, later = r.sample(range(61, 90), 2)
    shared = r.choice((False, True))
    binding = "b = a" if shared else "b = a[:]"
    kind = r.choice(("append", "replace", "pop"))
    index = r.randrange(len(vals))
    if kind == "append":
        edit, changed = f"b.append({extra})", vals + [extra]
    elif kind == "replace":
        edit = f"b[{index}] = {extra}"
        changed = vals[:index] + [extra] + vals[index + 1:]
    else:
        edit, changed = f"b.pop({index})", vals[:index] + vals[index + 1:]
    ans, wrong = (changed, vals) if shared else (vals, changed)
    code = f"a = {vals}\n{binding}\n{edit}\nprint(a)"
    link = "a 和 b 指向同一张列表" if shared else "b 是复制得到的新列表，可以和 a 分开修改"
    return _lesson("列表别名与复制", "判断列表修改是否能通过另一个名字看见", code, ans, wrong,
                   f"执行 {binding} 后，{link}。{edit} 将 b 改成 {changed}，此时 print(a) 输出 {ans}。",
                   f"混淆共享列表和复制列表，会把 a 的内容误判为 {wrong}。",
                   f"留意“{binding}”这行。通过 b 改动时，a 指向的是同一张列表吗？",
                   f"把最后的 print(a) 改为先执行 a.append({later})，再执行 print(b)。输出是什么？",
                   code.replace("print(a)", f"a.append({later})\nprint(b)", 1),
                   changed + [later] if shared else changed)


def _default_arg(r):
    unit, base, qty = r.randint(2, 15), r.randint(2, 18), r.randint(2, 14)
    override = r.choice([x for x in range(1, 20) if x != base])
    explicit = r.choice((True, False))
    mode = r.choice(("+", "-"))
    fname, arg = ("cost", "fee") if mode == "+" else ("bill", "discount")
    signature = f"def {fname}(qty, {arg}={base}):\n    return qty * {unit} {mode} {arg}\n"
    call = f"{fname}({qty}, {override})" if explicit else f"{fname}({qty})"
    actual = override if explicit else base
    sign = 1 if mode == "+" else -1
    ans, wrong = qty * unit + sign * actual, qty * unit + sign * (base if explicit else 0)
    code = signature + f"print({call})"
    next_call = f"{fname}({qty + 1}, {actual + 1})"
    return _lesson("函数的默认参数", "按调用内容确定函数实际使用的参数", code, ans, wrong,
                   f"调用 {call} 时 qty = {qty}，{arg} = {actual}。返回 {qty} × {unit} {mode} {actual} = {ans}。",
                   (f"传入的 {override} 优先于默认值 {base}，仍用默认值会算成 {wrong}。" if explicit else
                    f"调用时省略 {arg} 会使用默认值 {base}，将它当成 0 会算成 {wrong}。"),
                   f"调用处是否给了第二个参数？把实际的 qty 和 {arg} 代入 return。",
                   f"改为调用 {next_call} 并打印返回值，结果是什么？",
                   signature + f"print({next_call})", (qty + 1) * unit + sign * (actual + 1))


def _branch_order(r):
    div, other_div = r.sample(range(2, 10), 2)
    n = lcm(div, other_div) * r.randint(1, 12)
    cmp = r.choice(("mod", "threshold"))
    if cmp == "mod":
        first, second = f"n % {div} == 0", f"n % {other_div} == 0"
    else:
        first = f"n >= {n - r.randint(1, 4)}"
        second = f"n >= {n - r.randint(5, 9)}"
    a, b = r.sample(range(2, 75), 2)
    def program(c1, v1, c2, v2):
        return (f"n = {n}\nif {c1}:\n    points = {v1}\nelif {c2}:\n"
                f"    points = {v2}\nelse:\n    points = 0\nprint(points)")
    return _lesson("if 与 elif 的顺序", "条件重叠时只执行第一个成立的分支",
                   program(first, a, second, b), a, b,
                   f"n = {n}，{first} 与 {second} 都成立。先命中的 if 给 points 赋值 {a}，跳过 elif。",
                   f"把 elif 当成另一个独立的 if，就会用后面的 {b} 覆盖 {a}。",
                   "第一个条件若已成立，Python 还会进入同一串 elif 吗？",
                   "把前两个条件及各自的赋值语句互换位置，让原 elif 先判断，打印什么？",
                   program(second, b, first, a), b)


def _nested_pairs(r):
    # Include witnesses of both a matching and a non-matching pair by construction.
    mid = r.randint(12, 40)
    span = range(mid - 10, mid + 11)
    left = [mid] + r.sample([x for x in span if x != mid], r.randint(2, 4))
    right = [mid - 1, mid + 1] + r.sample([x for x in span if abs(x - mid) != 1], r.randint(1, 3))
    r.shuffle(left)
    r.shuffle(right)
    op = r.choice(("<", ">"))
    pairs = [(a, b) for a in left for b in right]
    # Sort-based lookup makes the reference count independent of the code's loop structure.
    ordered = sorted(right)
    counts = {a: len(right) - bisect_right(ordered, a) if op == "<" else
              bisect_left(ordered, a) for a in left}
    ans, wrong = sum(counts.values()), len(pairs)
    code = (f"left = {left}\nright = {right}\nhits = 0\nfor a in left:\n"
            f"    for b in right:\n        if a {op} b:\n            hits += 1\nprint(hits)")
    other = ">" if op == "<" else "<"
    other_count = len(pairs) - ans - sum(a == b for a, b in pairs)
    transfer = f"将条件 a {op} b 改为 a {other} b，会打印什么？"
    transfer_code = code.replace(f"a {op} b", f"a {other} b", 1)
    if other_count == ans:
        extra = max(left) + 1 if op == "<" else min(left) - 1
        transfer = f"把 {extra} 加到 right 末尾，条件保持 a {op} b，最后会打印什么？"
        transfer_code = code.replace(f"right = {right}", f"right = {right + [extra]}", 1)
        other_count = ans + len(left)
    rows = "，".join(f"a={a} 时 {counts[a]} 对" for a in left)
    return _lesson("嵌套循环配对", "为每个外层值统计符合条件的内层值", code, ans, wrong,
                   f"按 a {op} b 分别计数，{rows}。这些次数相加得到 {ans}。",
                   f"直接用 {len(left)} × {len(right)} 得到的是所有 {wrong} 对，其中包括不满足条件的。",
                   f"先固定 a = {left[0]}，right 中哪些 b 满足 a {op} b？换下一个 a 时呢？",
                   transfer, transfer_code, other_count)


def _loop_control(r):
    seq = r.sample(range(2, 65), r.randint(5, 8))
    mark_at = r.randint(1, len(seq) - 2)
    mark = seq[mark_at]
    action = r.choice(("break", "continue"))
    kept = seq[:mark_at] if action == "break" else seq[:mark_at] + seq[mark_at + 1:]
    alt = seq[:mark_at] + seq[mark_at + 1:] if action == "break" else seq[:mark_at]
    code = (f"values = {seq}\ntotal = 0\nfor n in values:\n    if n == {mark}:\n"
            f"        {action}\n    total += n\nprint(total)")
    new_index = mark_at + r.choice((-1, 1))
    new_mark = seq[new_index]
    transfer_kept = seq[:new_index] if action == "break" else seq[:new_index] + seq[new_index + 1:]
    meaning = "结束整个循环" if action == "break" else "仅跳过这一轮"
    return _lesson("break 与 continue", "分清结束循环和跳过一次迭代", code, sum(kept),
                   sum(alt), f"读到 {mark} 时，{action} 会{meaning}。实际加入总数的是 {kept}，和为 {sum(kept)}。",
                   f"把 break 和 continue 的作用弄反，就会把 {alt} 作为求和对象，得到 {sum(alt)}。",
                   f"读到 {mark} 的时候发生什么？后面的元素还有没有机会进入累加语句？",
                   f"把触发 {action} 的条件从 n == {mark} 改为 n == {new_mark}，其余不变，会打印什么？",
                   code.replace(f"n == {mark}:", f"n == {new_mark}:", 1), sum(transfer_kept))


def _recursion(r):
    step, floor, seed = r.randint(1, 4), r.randint(0, 5), r.randint(2, 15)
    depth, factor = r.randint(2, 5), r.randint(1, 4)
    n = floor + depth * step + r.randrange(step)
    chain = [n - k * step for k in range((n - floor + step - 1) // step)]
    end = chain[-1] - step
    ans = seed + factor * sum(chain)
    fn = r.choice(("score", "points", "collect"))
    term = "n" if factor == 1 else f"{factor} * n"
    code = (f"def {fn}(n):\n    if n <= {floor}:\n        return {seed}\n"
            f"    return {term} + {fn}(n - {step})\nprint({fn}({n}))")
    more = r.randint(1, 7)
    parts = [factor * k for k in chain]
    return _lesson("递归的停止条件", "沿调用链追踪递归返回值并计入终点", code, ans, ans - seed,
                   f"参数依次为 {chain + [end]}。到 {end} 时满足停止条件，返回 {seed}。"
                   f"沿途的贡献是 {parts}，加上这个 {seed} 得到 {ans}。",
                   f"到终点返回的是 {seed}，不是 0。漏掉它只算出 {ans - seed}。",
                   f"从 {n} 每次减 {step}，第一次到达不大于 {floor} 的数时函数返回多少？",
                   f"若停止分支从 return {seed} 改为 return {seed + more}，其余不变，会打印什么？",
                   code.replace(f"return {seed}\n", f"return {seed + more}\n", 1), ans + more)


def _enumerate(r):
    values = r.sample(range(2, 40), r.randint(3, 6))
    start = r.randint(1, 6)
    code = (f"values = {values}\ntotal = 0\nfor pos, val in enumerate(values, start={start}):\n"
            "    total += pos * val\nprint(total)")
    terms = [(start + i) * v for i, v in enumerate(values)]
    ans = sum(terms)
    wrong = ans - start * sum(values)
    new_start = r.randint(0, start - 1)
    return _lesson("enumerate 的起始编号", "同时追踪元素编号和元素值并计算加权和", code, ans, wrong,
                   f"首个编号是 {start}，末个是 {start + len(values) - 1}。"
                   f"编号乘相应元素的结果为 {terms}，求和得到 {ans}。",
                   f"忽略 start={start} 而从 0 编号，每个乘数都会少 {start}，结果变成 {wrong}。",
                   f"第一个 pos 对应的 val 是 {values[0]}。pos 是从哪个数开始的？",
                   f"把 enumerate 的 start 从 {start} 改为 {new_start}，最后打印什么？",
                   code.replace(f"start={start}", f"start={new_start}", 1),
                   ans + (new_start - start) * sum(values))


def _dictionary(r):
    keys = r.sample(_WORDS, 3)
    target, key2, key3 = keys
    units = r.sample(range(2, 34), r.randint(3, 5))
    entries = [(target, x) for x in units] + [(key2, r.randint(1, 9)), (key3, r.randint(2, 12))]
    r.shuffle(entries)
    vals = [v for k, v in entries if k == target]
    default = r.randint(0, 5)
    total, last = sum(vals) + default, vals[-1]
    code = (f"entries = {entries}\nsums = {{}}\nfor key, value in entries:\n"
            f"    sums[key] = sums.get(key, {default}) + value\nprint(sums[{target!r}])")
    replace_key = r.choice((key2, key3))
    contributions = [v for k, v in entries if k == replace_key]
    progress = []
    acc = default
    for v in vals:
        acc += v
        progress.append(acc)
    return _lesson("字典按键累加", "用 get 的缺省值开始，再按键积累记录", code, total, last,
                   f"键 {target!r} 第一次使用默认值 {default}。它依次收到 {vals}，"
                   f"累计值依次为 {progress}，最后打印 {total}。",
                   f"只保留 {target!r} 最后一次的 value 会得到 {last}，漏掉 get 取回的旧累计值。",
                   f"第一次遇到键 {target!r} 时 get 返回多少，第二次又返回多少？",
                   f"把最后一行改为 print(sums[{replace_key!r}])，会打印什么？",
                   code.replace(f"print(sums[{target!r}])", f"print(sums[{replace_key!r}])", 1),
                   default + sum(contributions))


def _zip_pairs(r):
    n = r.randint(3, 5)
    x = r.sample(range(2, 36), n)
    y = r.sample(range(3, 45), n)
    tail = r.randint(2, 30)
    longer = r.choice(("left", "right"))
    left = x + ([tail] if longer == "left" else [])
    right = y + ([tail] if longer == "right" else [])
    # The calculation uses addition or multiplication independently of leftover items.
    op = r.choice(("+", "*"))
    parts = [x[i] + y[i] if op == "+" else x[i] * y[i] for i in range(n)]
    ans = sum(parts)
    code = (f"left = {left}\nright = {right}\ntotal = 0\nfor a, b in zip(left, right):\n"
            f"    total += a {op} b\nprint(total)")
    added = r.randint(2, 12)
    shorter, prev = ("right", right) if longer == "left" else ("left", left)
    transfer_code = code.replace(f"{shorter} = {prev}", f"{shorter} = {prev + [added]}", 1)
    contribution = tail + added if op == "+" else tail * added
    imagined = 0 if op == "+" else 1
    return _lesson("zip 按位置配对", "识别不同长度列表中没有配对的末项", code, ans, ans + tail,
                   f"zip 只产生 {n} 对。逐对算 a {op} b 得 {parts}，相加为 {ans}。"
                   f"{longer} 的末项 {tail} 没有配对。",
                   f"误以为 zip 会用 {imagined} 补齐短列表，就会把多出的 {tail} 也算进去。",
                   f"较短的列表长度是多少？它用完后，{longer} 的末项还有搭档吗？",
                   f"在 {shorter} 末尾追加 {added}，其余不变，会打印什么？",
                   transfer_code, ans + contribution)


_GENERATORS = {
    "precedence": ("基础", _precedence), "division": ("基础", _division),
    "snapshot": ("基础", _snapshot), "slice": ("基础", _slice),
    "index": ("基础", _index), "boundary": ("基础", _boundary),
    "range_sum": ("进阶", _range_sum), "filter": ("进阶", _filter),
    "split_join": ("进阶", _split_join), "alias": ("进阶", _alias),
    "default_arg": ("进阶", _default_arg), "branch_order": ("进阶", _branch_order),
    "nested_pairs": ("挑战", _nested_pairs), "loop_control": ("挑战", _loop_control),
    "recursion": ("挑战", _recursion), "enumerate": ("挑战", _enumerate),
    "dictionary": ("挑战", _dictionary), "zip_pairs": ("挑战", _zip_pairs),
}

FAMILIES = {name: level for name, (level, _) in _GENERATORS.items()}


def make_case(family: str, rng: random.Random) -> dict:
    """返回一道原题、典型误答及可独立执行的迁移题。随机性只取自 rng。"""
    level, generator = _GENERATORS[family]
    return {"family": family, "level": level, **generator(rng)}
