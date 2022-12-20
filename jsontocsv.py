import json
import pandas as pd
from datetime import datetime


with open('output_path_test1.json', encoding='utf-8') as inputfile:
    df = json.load(inputfile)
    df = df["1"]["streams"]["ACCL"]["samples"]

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

with open('csvfile_test1.csv', 'w') as f:
    for i in range(len(lst)):
        for j in range(len(lst[i])):
            f.write(f"{lst[i][j]}")
            if j + 1 < len(lst[i]):
                f.write(f',')
            else:
                f.write('\n')
