import json
import pandas as pd
from datetime import datetime

all_paths = [   "GoPro/Timo Fietsen/Json_files/Timo_fietsen_GH010031",
                "GoPro/Timo Fietsen/Json_files/Timo_fietsen_GH010032",
                "GoPro/Timo Fietsen/Json_files/Timo_fietsen_GH010033",
                "GoPro/Timo Fietsen/Json_files/Timo_fietsen_GH010034",
                "GoPro/Timo Fietsen/Json_files/Timo_fietsen_GH010035",
                "GoPro/Timo Fietsen/Json_files/Timo_fietsen_GH020034"
                ]
desired_data = "GYRO" # can also be "GYRO"


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
