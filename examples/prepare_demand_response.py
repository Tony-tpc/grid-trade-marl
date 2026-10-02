"""python -m examples.prepare_demand_response --output runs/sn_mappo/data"""
import argparse
import json

from marl.envs.demand_response_data import prepare_opsd


def main() -> None:
    """解析脚本命令行并执行文档所述入口；输出写入显式指定的本地目录。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='runs/sn_mappo/data')
    args = parser.parse_args()
    print(json.dumps(prepare_opsd(args.output), indent=2))


if __name__ == '__main__':
    main()
