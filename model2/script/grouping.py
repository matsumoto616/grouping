# -*- coding: utf-8 -*-
import numpy as np
import pulp
import random
import itertools

class Solver:
    def __init__(self,problem,setting):
        # ステップ数
        self.step = 0

        # メンバー
        self.members = problem['members']
        self.priority = problem["priority"]
        self.number_of_people = len(self.members)
        self.people_indices = list(range(self.number_of_people))
        self.combi = list(itertools.combinations(self.people_indices,2)) # メンバーの組合せ（コンビ）

        # z_i：コンビiが一度でも同じグループになったことがあるとき1,そうでないとき0
        self.z = np.zeros(len(self.combi)) # z_iは0で初期化

        # グループ
        self.groups = problem['groups']
        self.number_of_group = len(self.groups)
        self.group_indices = list(range(self.number_of_group))

        # グループ毎の定員を決定（できるだけ平均化する，ライブラリとか組込関数とかでもっと綺麗にかけそう...）
        self.group_capacity = np.zeros(self.number_of_group)
        q, mod = divmod(self.number_of_people,self.number_of_group)
        for j in self.group_indices:
            if j < mod:
                self.group_capacity[j] = q + 1
            else:
                self.group_capacity[j] = q

        # ハイパーパラメータ
        self.WEIGHT_COEFFICIENT = setting['weight_coefficient']
        self.PRIORITY_WEIGHT_COEFFICIENT = setting['priority_weight_coefficient']


        # CBCオプション
        self.CBC_MSG = setting['cbc_msg']
        self.CBC_MAXSECONDS = setting['cbc_maxSeconds']
        
        # ランダム班分けのフラグ（比較用）
        self.IS_RANDOM = setting['is_random']

    def gen_group(self):
        '''
        [心臓部]
        前回までの班分けによって誰と一緒になったかの情報(z)を既知とし，次回の班分けを最適化する
        '''
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
            (self.people_indices,self.group_indices),
            0,
            1,
            pulp.LpBinary
        )

        # 決定変数y_ij：コンビi(メンバーの組合せ)がグループjに入るとき1,それ以外のとき0
        y = pulp.LpVariable.dicts(
            'y',
            (self.combi,self.group_indices),
            0,
            1,
            pulp.LpBinary
        )

        # w = pulp.LpVariable('w',0,self.number_of_people-1,pulp.LpInteger)

        #####################################################################################
        ###  制約条件
        #####################################################################################
        # (A) 全てのメンバーはどこかのグループに必ず入る
        for i in self.people_indices:
            prob += pulp.lpSum([x[i][j] for j in self.group_indices]) == 1

        # (B) グループ人数はgiven
        for j in self.group_indices:
            prob += pulp.lpSum([x[i][j] for i in self.people_indices]) == self.group_capacity[j]

        # (C) yはi1とi2がどちらもグループjなら1，そうでないとき0
        for j in self.group_indices:
            for (i1,i2) in self.combi:
                prob += 2*y[(i1,i2)][j] <= x[i1][j] + x[i2][j]
                prob += x[i1][j] + x[i2][j] <= y[(i1,i2)][j] + 1
                        
        #####################################################################################
        ###  目的関数
        #####################################################################################
        # 重みの計算（平準化を狙う）
        weight = self._calc_weight()

        # 一回も同じグループになったことのないコンビ(self.z=0)ができるだけ同じグループ(y=1)になるように目的関数を設定（これを最大化する）
        # 特に不遇な人（同じ人とばかりグループになっている人）が優先されるように重みづける(weight[i1]+weight[i2]+1)
        # ＋1しているのは重みの最小値が0にならないようにするため
        prob += pulp.lpSum([(weight[i1]+weight[i2]+1)*(1-self.z[c])*y[(i1,i2)][j]
                            for c, (i1,i2) in enumerate(self.combi) for j in self.group_indices])

        #####################################################################################
        ###  最適化実行
        #####################################################################################
        # 一回目はランダムに決定。is_random=Trueの場合は毎回ランダム（比較用）
        if self.step == 0 or self.IS_RANDOM:
            self._fix_random_group(x)

        # 最適化ソルバCBCで最適化実行
        prob.solve(pulp.PULP_CBC_CMD(path=None,
                                    keepFiles=1,
                                    mip=1,
                                    msg=self.CBC_MSG,
                                    cuts=None,
                                    presolve=None,
                                    strong=None,
                                    options=[],
                                    fracGap=None,
                                    maxSeconds=self.CBC_MAXSECONDS,
                                    threads=None))

        #####################################################################################
        ###  後処理
        #####################################################################################
        # zを更新
        for c, (i1,i2) in enumerate(self.combi):
            for j in self.group_indices:
                self.z[c] = max(self.z[c],y[(i1,i2)][j].value())    # 班分けで同じグループになったコンビのzを1にする

        # 結果出力
        self._gen_output(x,y)

        # ステップを進める
        self.step += 1

    def _fix_random_group(self,x):
        '''
        完全にランダムな班分けを行い，決定変数xを決定（1回目の班分けや比較用）
        '''
        res_members = self.members.copy()
        for j in self.group_indices:
            # グループ定員分だけランダムでメンバーを選択
            random_group = random.sample(res_members,k=int(self.group_capacity[j]))
            for member in random_group:
                x[self.members.index(member)][j].setInitialValue(1) # xの初期解を1に
                x[self.members.index(member)][j].fixValue()         # xを初期解で固定
                res_members.remove(member)                          # 班分けが終わったメンバーは削除

    def _calc_weight(self):
        '''
        できるだけ色々な人とグループになれるような，目的関数の重みを計算する
        （主役がいる場合などは，それを最優先に考慮する）←未実装
        '''
        # 各メンバーに対して，一緒のグループになったことがある人数を計算
        #（これが平準化されるように重みを決定する）
        num_met = np.zeros(self.number_of_people)
        for c, (i1,i2) in enumerate(self.combi):
            if self.z[c] == 1:
                num_met[i1] += 1
                num_met[i2] += 1

        # 最大値との差を重みとする（不遇な人ほど値が大きくなる）
        weight_tmp = num_met.max() - num_met

        # 重みを極端にすると，不遇な人を優先しすぎる恐れがあるので，気持ち小さく
        # この係数はハイパーパラメータとなる
        # さらに主役は優先度を高くする(PRIORITY∊{0,1},PRIORITY_WEIGHT_COEFFICIENT:大きい値)
        weight_tmp2 = [self.WEIGHT_COEFFICIENT + self.priority[i]*self.PRIORITY_WEIGHT_COEFFICIENT for i in range(self.number_of_people)]
        weight = [i*j for (i,j) in zip(weight_tmp,weight_tmp2)]

        return weight
 
    def _gen_output(self,x,y):
        '''
        結果の出力
        '''        
        print(f'\n##### grouping {self.step} #####')

        # 得られたグループ分けの出力
        for j in self.group_indices:
            print(f'Group {self.groups[j]}: {[self.members[i] for i in self.people_indices if x[i][j].value() == 1]}')

        # 各メンバーに対して，一緒のグループになったことがある人数を出力
        #（数値が大きい程よい，出来れば平準化されているほうがよい）
        num_met = np.zeros(self.number_of_people)
        for c, (i1,i2) in enumerate(self.combi):
            if self.z[c] == 1:
                num_met[i1] += 1
                num_met[i2] += 1
        print(f'num_met = {num_met}')

        # zの総和出力（大きいほど1回も一緒にならなかった組合せが少ない）
        print(f'z_sum = {self.z.sum()} / {len(self.combi)} (max)')