# README
## 環境構築
- pip install -r requirements.txt

## 実行方法
1. 問題ファイル作成（dat\problem.json）
    - members(振り分けたいメンバー)
    - groups(振り分け先の班)
    - number_of_steps(振り分けの回数)
    
    を設定

2. 設定ファイル作成（dat\setting.json）
    - is_random : false(最適化する), true(毎回ランダムで班分ける)
    - weight_coefficient : 平準化の強さ（0.01~0.1程度？）
    - cbc_msg : true(ソルバのログを出力), false(出力しない)
    - cbc_maxSeconds : 最適化ソルバの探索打ち切り時間[s]

3. 実行
    - python script\solver.py dat\problem.json dat\setting.json
    
    結果が標準出力に出力される。

## 出力の見方
- \#\#\#\#\# grouping i \#\#\#\#\#\# の直下にi回目の班分けが表示される
- Group i : ['A','B',… ] の形式でグループiのメンバーが表示される
- num_met : []の形式で，その回までに一緒になった人の数が表示される。数が大きい程よい，また，数のばらつきが少ない方がよい。並びはdat\problem.jsonのmembersに対応。
- z_sum : 全コンビの中で，一度は同じグループになったものの数。maxはメンバー数をpとしてpC2となる。

## 課題
- 20人ぐらいになると，計算時間が相当かかる。（100sの打ち切りでは実行可能解が得られないことも... その場合はsetting.jsonのcbc_maxSecondsを大きくする）
    - 15人ぐらいなら大丈夫
    - 対称性が悪さをしてそうなので，班分け前に何人か固定してしまってから最適化すればいけるかも
- 主人公補正に未対応。重みづけを工夫すればすぐできるはず。