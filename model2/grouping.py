import numpy as np
import pulp
import random
import itertools

# メンバー
members = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
members = members[:15]  # 人数を適当に... 15~20人ぐらいで回ればOK
number_of_people = len(members)
people_indices = list(range(number_of_people))
combi = list(itertools.combinations(people_indices,2)) # メンバーの組合せ

# グループ
groups = ['G1','G2','G3']   # 何班に分けるか
number_of_group = len(groups)
group_indices = list(range(number_of_group))

# グループ分け回数
number_of_times = 3 # zoomなら40分×3回=2時間程度なので3回か？
time_indices = list(range(number_of_times))

# グループ毎の定員を決定（できるだけ平均化する，ライブラリとか組込関数とかでもっと綺麗にかけそう...）
group_capacity = np.zeros(number_of_group)
q, mod = divmod(number_of_people,number_of_group)
for j in group_indices:
    if j < mod:
        group_capacity[j] = q + 1
    else:
        group_capacity[j] = q


def main():
    print('\n##### before optimization (random grouping) #####')
    # z_i：コンビiが一度でも同じグループになったことがあるとき1,そうでないとき0 (これを決定変数としたものがmodel1)
    # 0で初期化
    z0 = np.zeros(len(combi))
    
    # 各班分けごとに最適化し，zを更新していく
    for step in range(number_of_times):
        z = gen_group(z0,step,is_random=True)   # 比較用に毎回ランダムで班分け
        z0 = z 

    print('\n##### After optimization #####')
    # z_i：コンビiが一度でも同じグループになったことがあるとき1,そうでないとき0 (これを決定変数としたものがmodel1)
    # 0で初期化
    z0 = np.zeros(len(combi))
    
    # 各班分けごとに最適化し，zを更新していく
    for step in range(number_of_times):
        z = gen_group(z0,step,is_random=False)
        z0 = z 

# 心臓部
# 前回までの班分けによって誰と一緒になったかの情報(z)を既知とし，次回の班分けを決定する
def gen_group(z,step,is_random):

    #####################################################################################
    ###  線形計画問題インスタンス生成（本問題は0-1整数計画になる）
    #####################################################################################
    prob = pulp.LpProblem(sense=pulp.LpMaximize)

    #####################################################################################
    ###  決定変数
    #####################################################################################
    # 決定変数x_ij：メンバーiがグループjに入るとき1,それ以外のとき0
    x = pulp.LpVariable.dicts(
        'x',
        (people_indices,group_indices),
        0,
        1,
        pulp.LpBinary
    )

    # 決定変数y_ij：コンビi(メンバーの組合せ)がグループjに入るとき1,それ以外のとき0
    y = pulp.LpVariable.dicts(
        'y',
        (combi,group_indices),
        0,
        1,
        pulp.LpBinary
    )

    # w = pulp.LpVariable('w',0,number_of_people-1,pulp.LpInteger)

    #####################################################################################
    ###  制約条件
    #####################################################################################
    # 全てのメンバーはどこかのグループに必ず入る
    for i in people_indices:
        prob += pulp.lpSum([x[i][j] for j in group_indices]) == 1

    # グループ人数はgiven
    for j in group_indices:
        prob += pulp.lpSum([x[i][j] for i in people_indices]) == group_capacity[j]

    # yはi1とi2がどちらもグループjなら1，そうでないとき0
    for j in group_indices:
        for (i1,i2) in combi:
            prob += 2*y[(i1,i2)][j] <= x[i1][j] + x[i2][j]
            prob += x[i1][j] + x[i2][j] <= y[(i1,i2)][j] + 1
                    
    #####################################################################################
    ###  目的関数
    #####################################################################################
    # 重みの計算（平準化を狙う）
    weight = calc_weight(z)

    # 一回も同じグループになったことのないコンビ(z=0)ができるだけ同じグループ(y=1)になるように目的関数を設定（これを最大化する）
    # 特に不遇な人（同じ人とばかりグループになっている人）が優先されるように重みづける(weight[i1]+weight[i2]+1)
    # ＋1しているのは重みの最小値が0にならないようにするため
    prob += pulp.lpSum([(weight[i1]+weight[i2]+1)*(1-z[c])*y[(i1,i2)][j] for c, (i1,i2) in enumerate(combi) for j in group_indices])


    #####################################################################################
    ###  最適化実行
    #####################################################################################
    # 一回目はランダムに決定。is_random=Trueの場合は毎回ランダム（比較用）
    if step == 0 or is_random:
        fix_random_group(x,group_capacity)

    # 最適化ソルバCBCで最適化実行
    prob.solve(pulp.PULP_CBC_CMD(path=None,
                                keepFiles=1,
                                mip=1,
                                msg=0,
                                cuts=None,
                                presolve=None,
                                strong=None,
                                options=[],
                                fracGap=None,
                                maxSeconds=100,
                                threads=None))

    #####################################################################################
    ###  後処理
    #####################################################################################
    # zを更新
    for c, (i1,i2) in enumerate(combi):
        for j in group_indices:
            z[c] = max(z[c],y[(i1,i2)][j].value())

    # 結果出力
    gen_output(x,y,z,step)

    return z

# 完全にランダムな班分けを行う（1回目の班分けや比較用）
def fix_random_group(x,group_capacity):
    res_members = members.copy()
    for j in group_indices:
        random_group = random.sample(res_members,k=int(group_capacity[j]))
        for member in random_group:
            x[members.index(member)][j].setInitialValue(1)
            x[members.index(member)][j].fixValue()
            res_members.remove(member)

# できるだけ色々な人とグループになれるような，目的関数の重みを計算する
# （主役がいる場合などは，それを最優先に考慮する）←未実装
def calc_weight(z):
    # 各メンバーに対して，一緒のグループになったことがある人数を計算
    #（これが平準化されるように重みを決定する）
    num_met = np.zeros(number_of_people)
    for c, (i1,i2) in enumerate(combi):
        if z[c] == 1:
            num_met[i1] += 1
            num_met[i2] += 1

    # 最大値との差を重みとする（不遇な人ほど値が大きくなる）
    weight = num_met.max() - num_met

    # 重みを極端にすると，不遇な人を優先しすぎる恐れがあるので，気持ち小さく
    weight *= 0.01

    return weight


# 1回目は辞書的にグループを決定（必要に応じてmembersリストをランダムシャッフルしてもよい）
def fix_initial_group(x,group_capacity):
    count_global = 0   
    for j in group_indices:
        count = 0
        for i in people_indices[count_global:]:
            if count < group_capacity[j]:
                x[i][j].setInitialValue(1)
                x[i][j].fixValue()
                count += 1
            else:
                x[i][j].setInitialValue(0)
                x[i][j].fixValue()
        count_global += count

# 結果の出力
def gen_output(x,y,z,step):
    
    print(f'\n##### grouping {step} #####')

    # 得られたグループ分けの出力
    for j in group_indices:
        print(f'Group {groups[j]}: {[members[i] for i in people_indices if x[i][j].value() == 1]}')

    # 各メンバーに対して，一緒のグループになったことがある人数を出力
    #（数値が大きい程よい，出来れば平準化されているほうがよい）
    num_met = np.zeros(number_of_people)
    for c, (i1,i2) in enumerate(combi):
        if z[c] == 1:
            num_met[i1] += 1
            num_met[i2] += 1
    print(num_met)

    # zの総和出力（大きいほど1回も一緒にならなかった組合せが少ない）
    print(f'z_sum = {z.sum()} / {len(combi)} (max)')

if __name__ == '__main__':
    main()