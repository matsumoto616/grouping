# -*- coding: utf-8 -*-
import json
import argparse
import grouping

###############################################################################
def main():
    # コマンドライン引数
    parser = argparse.ArgumentParser(
        description='A script to obtain an approximate solution for grouping problem.')
    parser.add_argument('problem_file',
                        help='specify the problem file.',
                        type=str)
    parser.add_argument('setting_file',
                        help='specify the setting file.',
                        type=str)
    args = parser.parse_args()

    # 問題のインポート
    with open(args.problem_file, 'r') as f:
        problem = json.load(f)

    # 設定のインポート
    with open(args.setting_file, 'r') as f:
        setting = json.load(f)
    
    # インスタンス生成
    solver = grouping.Solver(problem,setting)

    # 班分け
    for step in range(problem['number_of_steps']):
        solver.gen_group()

###############################################################################
if __name__ == "__main__":
    main()

###############################################################################
# END
###############################################################################
