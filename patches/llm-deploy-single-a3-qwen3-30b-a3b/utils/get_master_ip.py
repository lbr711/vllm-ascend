import json
import sys
from pathlib import Path



GRT_PATH = Path("/user/global/config/..data/global_rank_table.json")


def get_master_ip(grt_path: Path = GRT_PATH) -> str:
    with grt_path.open(encoding="utf-8") as f:
        data = json.load(f)
    return data["server_group_list"][0]["server_list"][0]["server_ip"]


def get_all_hosts(grt_path: Path = GRT_PATH) -> list[str]:
    with grt_path.open(encoding="utf-8") as f:
        data = json.load(f)
    hosts = []
    for group in data.get("server_group_list", []):
        for server in group.get("server_list", []):
            if server.get("device"):
                hosts.append(server["server_ip"])
    return hosts


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", default=GRT_PATH, type=Path)
    parser.add_argument("--all", action="store_true", dest="all_hosts",
                        help="print all host IPs (servers with device)")
    args = parser.parse_args()

    if args.all_hosts:
        print(" ".join(get_all_hosts(args.path)))
    else:
        print(get_master_ip(args.path))
