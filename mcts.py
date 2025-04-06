import numpy
from node import treeNode
# from expand import expand
# from backpropagate import back_propagate
import math
import random
import time
def isTerminal(node, mcts_task):
    if mcts_task.reward_model_type == 'vm':
        return node.V >= 9.3
    else:
        return False
    
def getBestChild(node, mcts_task):
    bestValue = mcts_task.low
    bestNodes = []
    for child in node.children.values():
        nodeValue = child.V + mcts_task.exploration_constant * math.sqrt(
            2 * math.log(node.numVisits) / child.numVisits) if child.numVisits > 0 else child.V + mcts_task.INF
        if nodeValue > bestValue:
            bestValue = nodeValue
            bestNodes = [child]
        elif nodeValue == bestValue:
            bestNodes.append(child)
    return random.choice(bestNodes)

def selectNode(node, mcts_task):
    while node.isFullyExpanded:
        bestValue = mcts_task.low
        bestNodes = []
        for child in node.children.values():
            nodeValue = child.V + mcts_task.exploration_constant * math.sqrt(
                2 * math.log(node.numVisits) / child.numVisits) if child.numVisits > 0 else child.V + mcts_task.INF
            if nodeValue > bestValue:
                bestValue = nodeValue
                bestNodes = [child]
            elif nodeValue == bestValue:
                bestNodes.append(child)
        node = random.choice(bestNodes)
    if isTerminal(node, mcts_task):
        node.final_ans_flag = 1
        return True, node
    else:
        return False, node

def expand(node: treeNode, mcts_task):
    if not node.reflection:
        if mcts_task.use_reflection == 'common': #simple
            reflection = mcts_task.get_reflection(node.y, node.depth + 1)
        else:  # simple
            reflection = mcts_task.get_simple_reflection(node.y, node.depth + 1)
        node.update_reflection(reflection)
    if node.reflection == '<end>':
        return node
    actions = get_next_steps_expand(node, mcts_task)
    if not actions:
        node.update_reflection('<end>')
        return node

    for action in actions:
        if action not in node.children.keys():
            node.append_children(action)
            child = node.children[action]
            value = mcts_task.get_step_value(child.y,action)
            child.update_value(value)
            if mcts_task.sample_value == 'full':
                if mcts_task.use_reflection == 'common':
                    child.update_reflection(mcts_task.get_reflection(child.y, child.depth + 1))
                else:
                    child.update_reflection(mcts_task.get_simple_reflection(child.y, child.depth + 1))
            child.visit_sequence = mcts_task.node_count
            mcts_task.update_count()
    node.isFullyExpanded = True
    return node

def get_next_steps_expand(node: treeNode, mcts_task):
    next_steps = []
    reflection = node.reflection
    for i in range(int(mcts_task.branch)):
        proposal = ''
        cnt = 3
        while not proposal and cnt:
            if node.y == '':
                proposal = mcts_task.get_first_step(node.y)
            else:
                proposal = mcts_task.get_next_step(node.y, node.depth + 1)
            cnt -= 1
        if not proposal:
            continue
        next_steps.append(proposal)
    return next_steps

def back_propagate(node):
    while node is not None:
        node.numVisits += 1
        if node.isFullyExpanded:
            child_Vs = [child.V * child.numVisits for child in node.children.values()]
            total_num_visits = sum([child.numVisits for child in node.children.values()])
            if total_num_visits > 0:
                node.V = sum(child_Vs) / total_num_visits
        node = node.parent

def MCTS(mcts_task):
    root, node, finish = MCTS_search(mcts_task)

    if mcts_task.sample_value == 'full':
        print('표본 추출완료。\n')
        return None, -1, root
    else:
        if finish is not None:
            print(f'최종 해결책을 찾았습니다 !\nSolution:{node.y}\n')
            return node, finish, root

        else:
            best_node, best_V = root.getBestV()
            print(f'규정된 시간, iteration 내에 요구되는 값을 만족하는 해결책을 찾지 못해 최고가치 해로 대체한다。\nSolution:{best_node.y}\n')
            return best_node, -1, root

def MCTS_search(mcts_task):
    root = treeNode('')
    if mcts_task.limit_type == 'time':
        timeLimit = time.time() + mcts_task.time_limit / 1000
        time_start = time.time()
        while time.time() < timeLimit:
            print(f'<새로운 탐색 라운드 시작, 현재 총 시간:{time.time() - time_start}>\n') #开始新搜索轮次，目前总时间

            flag, node, root = executeRound(root, mcts_task)
            if flag:
                print('해결책을 찾았습니다！\n')#已找到解决方案
                return root, node, time.time() - time_start
    else:
        for i in range(mcts_task.iteration_limit):
            print(f'<새로운 탐색 라운드 시작, 현재 완료된 라운드 수:{i}>\n')
            flag, node, root = executeRound(root, mcts_task)
            if flag:
                print('해결책을 찾았습니다！\n')
                return root, node, i + 1
    return root, None, None

def executeRound(root, mcts_task):
    # execute a selection-expansion-simulation-backpropagation round

    print('-' * 40)
    print('selection phase\n')
    flag, node = selectNode(root, mcts_task) #if isTerminal(node, mcts_task)이면 node.final_ans_flag = 1,return True, node
    print(f'{flag}')
    if flag:
        if mcts_task.sample_value != 'full':
            return True, node, root
        else:
            node.reflection = '<end>'

    print('-' * 40)
    print('Expansion Phase\n')
    if node.reflection == '<end>':
        print('이 단계를 건너뜁니다.\n')
    else:
        node = expand(node, mcts_task)

    if mcts_task.reward_model_type == 'vm':
        print('-' * 40)
        print('Simulation Search Phase\n')
        if node.reflection == '<end>':
            print('이 단계를 건너뜁니다.\n')
        else:
            roll_node = getBestChild(node, mcts_task)
            best_V = greedyPolicy(roll_node, mcts_task)
            #  if mcts_task.roll_policy == 'greedy' else randomPolicy(roll_node,mcts_task)
            roll_node.V = roll_node.V * (1 - mcts_task.alpha) + best_V * mcts_task.alpha
            roll_node.numVisits += 1

    print('-' * 40)
    print('Backpropagation Phase\n')
    back_propagate(node)
    return False, node, root

def greedyPolicy(node: treeNode, mcts_task):
    max_V = mcts_task.low
    strs = node.y
    cur_step = node.depth + 1
    if mcts_task.use_reflection == 'common':
        reflection = mcts_task.get_reflection(strs, cur_step)
    else:
        reflection = mcts_task.get_simple_reflection(strs, cur_step)
    node.update_reflection(reflection)
    if reflection == '<end>':
        print('This step has been resolved and does not require simulation.\n')
        return node.V
    for i in range(mcts_task.roll_forward_steps): # 디폴트 - 3
        actions = get_next_steps_roll(strs, cur_step, mcts_task)  # str_list
        if not actions:
            break
        new_ys = [strs + action for action in actions]
        cur_step += 1
        values = [mcts_task.get_step_value(new_y, action) for new_y in new_ys for action in actions]
        idx = numpy.argmax(values)
        strs = new_ys[idx]
        value = values[idx]
        if value > max_V:
            max_V = value
        if mcts_task.use_reflection == 'common':
            cur_ref = mcts_task.get_reflection(strs, cur_step)
        else:
            cur_ref = mcts_task.get_simple_reflection(strs, cur_step)
        if cur_ref == '<end>':
            break
    return max_V

def get_next_steps_roll(y: str, step_n: int, mcts_task):
     next_steps = []
     for i in range(mcts_task.roll_branch): #디폴트 roll_branch=1
         proposal = ''
         cnt = 3
         while not proposal and cnt:
            if y == '':
                proposal = mcts_task.get_first_step(y)
            else:
                proposal = mcts_task.get_next_step(y, step_n) #다음단계 생성하는 함수
            cnt -= 1
         if not proposal:
             continue
         next_steps.append(proposal)
     return next_steps