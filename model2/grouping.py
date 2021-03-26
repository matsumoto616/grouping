import numpy as np
import pulp
import itertools

# メンバー
members = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
#members = ['A','B','C','D','E','F','G','H','I','J','K']
number_of_people = len(members)
people_indices = list(range(number_of_people))
combi_indices = list(itertools.combinations(people_indices,2)) # メンバーの組合せ

# グループ
groups = ['G1','G2','G3']
number_of_group = len(groups)
group_indices = list(range(number_of_group))

# グループ分け回数
number_of_times = 3
time_indices = list(range(number_of_times))

# グループ毎の人数を決定（できるだけ平均化する，もっと上手く書けない？）
group_capacity = np.zeros(number_of_group)
q, mod = divmod(number_of_people,number_of_group)
for j in group_indices:
    if j < mod:
        group_capacity[j] = q + 1
    else:
        group_capacity[j] = q

def main():
    # z_i：コンビiが一度でも同じグループになったことがあるとき1,そうでないとき0 (これを決定変数としたものがmodel1)
    # 0で初期化
    z0 = np.zeros(len(combi_indices))
    
    # 各班分けごとに最適化し，zを更新していく
    for step in range(number_of_times):
        z = solve(z0,step)
        z0 = z 

# 前回までの班分けによって誰と一緒になったかの情報(z)を既知とし，次回の班分けを決定する
def solve(z,step):

    # 問題インスタンス生成
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
        (combi_indices,group_indices),
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
        for (i1,i2) in combi_indices:
            prob += 2*y[(i1,i2)][j] <= x[i1][j] + x[i2][j]
            prob += x[i1][j] + x[i2][j] <= y[(i1,i2)][j] + 1
                    
    #####################################################################################
    ###  目的関数
    #####################################################################################
    # 一回も同じグループになったことのないコンビ(z=0)ができるだけ同じグループ(y=1)になるように目的関数を設定（これを最大化する）
    prob += pulp.lpSum([(1-z[c])*y[(i1,i2)][j] for c, (i1,i2) in enumerate(combi_indices) for j in group_indices])

    # 一回目は辞書的に決定
    if step == 0:
        fix_initial_group(x,group_capacity)

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

    # zを更新
    for c, (i1,i2) in enumerate(combi_indices):
        for j in group_indices:
            z[c] = max(z[c],y[(i1,i2)][j].value())

    # 結果出力
    output_group(x,y,z)

    return z

# 1回目は辞書的にグループを決定
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
def output_group(x,y,z):
    # zの総和出力（大きいほど1回も一緒にならなかった組合せが少ない）
    print(f'z_sum = {z.sum()} / {len(combi_indices)} (max)')

    # グループの出力
    for j in group_indices:
        print(f'Group {groups[j]}: {[members[i] for i in people_indices if x[i][j].value() == 1]}')
    # for c, (i1,i2) in enumerate(combi_indices):


if __name__ == '__main__':
    main()