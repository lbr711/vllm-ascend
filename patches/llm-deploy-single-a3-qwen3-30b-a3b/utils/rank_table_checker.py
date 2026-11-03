import json
import os
import time
from datetime import datetime

GLOBAL_RANK_TABLE_ENV = 'GLOBAL_RANK_TABLE_FILE_PATH'

def wait_and_get_completed_global_rank_table():
    global_rank_table_path = os.getenv(GLOBAL_RANK_TABLE_ENV)
    if not global_rank_table_path:
        print('read env \"{}\" failed'.format(GLOBAL_RANK_TABLE_ENV))
        return

    while True:
        try:
            rank_table = get_rank_table(global_rank_table_path)
            if rank_table is not None:
                return rank_table
        except Exception as e:
            print(e)
        time.sleep(1)

def get_rank_table(global_rank_table_path=None):
    if global_rank_table_path is None:
        global_rank_table_path = os.getenv(GLOBAL_RANK_TABLE_ENV)
    with open(global_rank_table_path, 'r') as file:
        buf = file.read()
    rank_table = json.loads(buf)
    if rank_table["status"] == "completed":
        return rank_table
    else:
        date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"{date} status of rank table is not completed")
        return None


if __name__ == "__main__":
    wait_and_get_completed_global_rank_table()