
def llm_verify(ans, real_ans, judge_model='gpt-4-1106-preview'):
    prompt = '下面将输入两段文字，第一段文字为某道理科题目的一个解答或答案（不一定正确），第二段是这道题目的标准答案。请判断第一段解答得到的答案与标准答案在数学意义上是否一致，并根据判断直接输出‘0’或’1‘，不需要输出任何别的信息。如果答案一致，请输出‘1’；否则，只要答案不匹配，或者第一个文段中没有明确指出答案也没有输出latex表达式，请输出‘0’；如果第一段解答与标准答案之间关系模糊，请输出‘0’。\n'
    qry = prompt + '文段1:' + ans + '\n' + '文段2:' + real_ans + '\n输出:'
    lbl = ''
    cnt = 5
    while lbl == '' and cnt:
        out = ''
        try:
            chat_comp = openai.ChatCompletion.create(model=judge_model, messages=[{"role": "user", "content": qry}])
            out = chat_comp.choices[0].message.content[0]
        except Exception as e:
            print(f'Error:{e}\n')
        if out == '0' or out == '1':
            lbl = out
        else:
            cnt -= 1
    if not cnt:
        return 0
    return int(lbl)
def extract_summary_from_solution(solution: str):
    pattern = r"\\boxed\{(.*)\}"
    match = re.findall(pattern, solution)
    if match:
        summary = 'The final answer is ' + match[-1]
    elif '####' in solution:
        extracted = solution.split('####')[-1].strip()
        if len(extracted) > 1:
            if extracted[-1] == '.':
                extracted = extracted[:-1].strip()
            if len(extracted) > 1:
                if extracted[0] == ':':
                    extracted = extracted[1:].strip()
        summary = 'The final answer is ' + extracted
    elif 'The final answer is' in solution:
        extracted = solution.split('The final answer is')[-1].strip()
        if len(extracted) > 1:
            if extracted[-1] == '.':
                extracted = extracted[:-1].strip()
            if len(extracted) > 1:
                if extracted[0] == ':':
                    extracted = extracted[1:].strip()
        summary = 'The final answer is ' + extracted
    elif 'The answer is' in solution:
        extracted = solution.split('The answer is')[-1].strip()
        if len(extracted) > 1:
            if extracted[-1] == '.':
                extracted = extracted[:-1].strip()
            if len(extracted) > 1:
                if extracted[0] == ':':
                    extracted = extracted[1:].strip()
        summary = 'The final answer is ' + extracted
    elif 'final answer is' in solution:
        extracted = solution.split('final answer is')[-1].strip()
        if len(extracted) > 1:
            if extracted[-1] == '.':
                extracted = extracted[:-1].strip()
            if len(extracted) > 1:
                if extracted[0] == ':':
                    extracted = extracted[1:].strip()
        summary = 'The final answer is ' + extracted
    elif 'answer is' in solution:
        extracted = solution.split('answer is')[-1].strip()
        if len(extracted) > 1:
            if extracted[-1] == '.':
                extracted = extracted[:-1].strip()
            if len(extracted) > 1:
                if extracted[0] == ':':
                    extracted = extracted[1:].strip()
        summary = 'The final answer is ' + extracted
    else:
        summary = ''
    print('Extracted summary: ', summary, '\n')
    return summary
def exact_match_score(prediction, ground_truth): #summ, self.answer
    prediction = extract_answer(prediction)
    return grade_answer(prediction, ground_truth)

def grade_answer(given_answer: str, ground_truth: str) -> bool:
    """
    The answer will be considered correct if:
    (a) it normalizes to the same string as the ground truth answer
    OR
    (b) sympy can simplify the difference between the expressions to 0
    """
    if given_answer is None:
        return False

    ground_truth_normalized_mathd = normalize_answer(ground_truth)
    given_answer_normalized_mathd = normalize_answer(given_answer)

    # be at least as lenient as mathd
    if ground_truth_normalized_mathd == given_answer_normalized_mathd:
        return True

    ground_truth_normalized = _normalize(ground_truth)
    given_normalized = _normalize(given_answer)

    if ground_truth_normalized is None:
        return False

    if ground_truth_normalized == given_normalized:
        return True

    if len(given_normalized) == 0:
        return False

    ground_truth_elems = split_tuple(ground_truth_normalized)
    given_elems = split_tuple(given_normalized)

    if len(ground_truth_elems) > 1 and (
            ground_truth_normalized[0] != given_normalized[0]
            or ground_truth_normalized[-1] != given_normalized[-1]
    ):
        is_correct = False
    elif len(ground_truth_elems) != len(given_elems):
        is_correct = False
    else:
        is_correct = True
        for ground_truth_elem, given_elem in zip(ground_truth_elems, given_elems):
            if _is_frac(ground_truth_elem) and _is_frac(given_elem):
                # if fractions aren't reduced, then shouldn't be marked as correct
                # so, we don't want to allow sympy.simplify in this case
                is_correct = ground_truth_elem == given_elem
            elif _is_float(ground_truth_elem) and _is_float(given_elem):
                # if the ground truth answer is an integer, we require the given answer to be a strict match (no sympy.simplify)
                if _str_is_int(ground_truth_elem):
                    try:
                        is_correct = round(float(given_elem)) == int(ground_truth_elem)
                    except:
                        is_correct = False
                else:
                    ground_truth_elem = float(ground_truth_elem)
                    given_elem = float(given_elem)
                    eps = abs(ground_truth_elem) * 0.04
                    if ground_truth_elem - eps <= given_elem <= ground_truth_elem + eps:
                        is_correct = True
                    else:
                        is_correct = False
            else:
                is_correct = are_equal_under_sympy(ground_truth_elem, given_elem)
            if not is_correct:
                break

    return is_correct
def are_equal_under_sympy(ground_truth_normalized: str, given_normalized: str):
    are_equal = False
    try:
        expr = f"({ground_truth_normalized})-({given_normalized})"
        if should_allow_eval(expr):
            sympy_diff = _sympy_parse(expr)
            simplified = sympy.simplify(sympy_diff)
            if abs(simplified) <= 0.04 * sympy.simplify(ground_truth_normalized):
                are_equal = True
    except:
        pass
    return are_equal

def verify_float(answer: float, output: str):
    if not output:
        print(f'The output is empty and cannot match the answer!\n')
        return False

    if '综上所述，' in output:
        spl_ans = output.split('综上所述，')[-1]
        spl_ans = spl_ans.strip()
    else:
        spl_ans = output.strip()

    try:
        match = re.findall(r'[^^{.\-0123456789]-?[0-9]+\.?[0-9]*[^^}.0123456789]', spl_ans)[-1][1:][:-1]
        model_ans = float(match)

        # standard (adjustable)
        if abs(answer) >= 1:
            result = math.isclose(model_ans, answer, abs_tol=0.1)
        else:
            result = math.isclose(model_ans, answer, rel_tol=0.1)

        print(f'The ans of model is:{model_ans}, while the ground truth is {answer}.\n')
        return result

    except Exception as e:
        try:
            match = re.findall(r'-?[0-9]+\.?[0-9]*', spl_ans)[-1]
            model_ans = float(match)

            # standard (adjustable)
            if abs(answer) >= 1:
                result = math.isclose(model_ans, answer, abs_tol=0.1)
            else:
                result = math.isclose(model_ans, answer, rel_tol=0.1)

            print(f'The ans of model is:{model_ans}, while the ground truth is {answer}.\n')
            return result
        except Exception as e:
            print(f'Result not matched, error type:{e}\n')
            print(f'The ans of model is:{spl_ans}, while the ground truth is {answer}.\n')

            return False