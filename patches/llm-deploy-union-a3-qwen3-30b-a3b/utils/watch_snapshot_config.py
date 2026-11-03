import argparse
import json
import os
import re
import subprocess
import time
import tempfile

FLAG_FILE = "/root/.grusflag"
GRT_PATH = "/user/global/config/..data/global_rank_table.json"
CHECK_INTERVAL = 5


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-metadata-path", type=str, default="/home/code/conf/snapshot_metadata.json")
    return parser.parse_args()


def read_grt_status():
    try:
        with open(GRT_PATH, "r") as f:
            data = json.load(f)
        return data.get("status"), data
    except Exception:
        return None, None


def get_eth0_ip():
    try:
        output = subprocess.check_output(["ip", "addr", "show", "eth0"], text=True)
        match = re.search(r"inet\s+([\d.]+)", output)
        if match:
            return match.group(1)
    except Exception:
        pass
    return None


def find_first_valid_server_ip(grt_data):
    groups = grt_data.get("server_group_list", [])
    for group in groups:
        servers = group.get("server_list", [])
        for server in servers:
            if server.get("device"):
                return server.get("server_ip")
    return None


def resolve_master_ip(grt_data):
    is_master = os.environ.get("IS_MASTER", "0") == "1"
    eth0_ip = get_eth0_ip()
    if is_master:
        return eth0_ip
    # Non-master: find the server_ip from grt that differs from eth0
    groups = grt_data.get("server_group_list", [])
    for group in groups:
        servers = group.get("server_list", [])
        for server in servers:
            if server.get("device") and server.get("server_ip") != eth0_ip:
                return server["server_ip"]
    return find_first_valid_server_ip(grt_data)


def read_existing_metadata(snapshot_metadata_path):
    try:
        with open(snapshot_metadata_path, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def atomic_write_json(path, data):
    dir_name = os.path.dirname(path) or "."
    fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f)
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def main():
    args = parse_args()
    snapshot_metadata_path = args.snapshot_metadata_path
    threshold  = 3
    cnt = 0
    while True:
        if os.path.exists(FLAG_FILE):
            status, grt_data = read_grt_status()
            if status == "completed" and grt_data:
                ip = resolve_master_ip(grt_data)
                if ip:
                    current_content = read_existing_metadata(snapshot_metadata_path)
                    model_save_path = current_content.get("model_save_path")
                    new_data = {"data_parallel_master_ip": ip}
                    if model_save_path:
                        new_data["model_load_path"] = model_save_path
                        new_data["model_save_path"] = model_save_path
                    if current_content.get("data_parallel_master_ip") != ip:
                        cnt += 1
                        if cnt > threshold:
                            print(f"write new snapshot_metadata: {new_data}")
                            atomic_write_json(snapshot_metadata_path, new_data)
                            cnt = 0
                else:
                    print("Error: Failed to obtain a valid server_ip from global_rank_table.json")
        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    main()