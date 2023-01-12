import json
import pandas as pd
from datetime import datetime

all_paths = [   "GoPro/Ian_stofzuigen/Json_files/output_path_gang_langestrook_2",
                "GoPro/Ian_stofzuigen/Json_files/output_path_keuken_langestrook",
                "GoPro/Ian_stofzuigen/Json_files/output_path_slaapkamer_alles",
                "GoPro/Ian_stofzuigen/Json_files/output_path_slaapkamer_onder_bed",
                "GoPro/Ian_stofzuigen/Json_files/output_path_gang_langestrook",
                "GoPro/Ian_stofzuigen/Json_files/output_path_trap_2",
                "GoPro/Ian_stofzuigen/Json_files/output_path_slaapkamer_langestrook",
                "GoPro/Ian_stofzuigen/Json_files/output_path_trap",
                "GoPro/Ian_stofzuigen/Json_files/output_path_voorkamer_bank",
                "GoPro/Ian_stofzuigen/Json_files/output_path_voorkamer_lange_stroken",
                "GoPro/Ian_stofzuigen/Json_files/output_path_voorkamer_tapijt"
                ]
desired_data = "ACCL" # can also be "GYRO"


for path in all_paths:


    with open(f'{path}.json', encoding='utf-8') as inputfile:
        df = json.load(inputfile)
        df = df["1"]["streams"][desired_data]["samples"]

        lst: list[list[str | float]] = []
        lst.append([])
        dct = df[0]
        time0 = dct["date"]
        time0 = time0[11:-1]

        time0 = datetime.strptime(time0, '%H:%M:%S.%f')
        lst[0].append((time0-time0).total_seconds())
        for j in range(len(dct["value"])):
            lst[0].append(dct["value"][j])

        for i in range(1, len(df)):
            dct = df[i]
            lst.append([])
            time = dct["date"]
            time = time[11:-1]
            time = datetime.strptime(time, '%H:%M:%S.%f')
            lst[i].append((time-time0).total_seconds())
            for j in range(len(dct["value"])):
                lst[i].append(dct["value"][j])

    with open(f'{path}-{desired_data}.csv', 'w') as f:
        for i in range(len(lst)):
            for j in range(len(lst[i])):
                f.write(f"{lst[i][j]}")
                if j + 1 < len(lst[i]):
                    f.write(f',')
                else:
                    f.write('\n')
