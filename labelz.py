import pandas as pd
import os
import numpy as np

table = pd.read_csv("../../data/gravo/table_data.csv")
# print(len(table[(table["NCCT"] == "OK") & (table["rankin-23"] != "None")]))

out = {"rankin": [], "patient_id": [], "binary_rankin": [], "set": []}
zeors = 0
ones = 0
hmm = 0
for _, row in table.iterrows():
    r = row["rankin-23"]
    if (r != "None") and r.isnumeric():
        r = int(r)
        out["rankin"].append(r)
        out["patient_id"].append( row["idProcessoLocal"] )
        out["binary_rankin"].append( int(r > 2) )
        if r > 2:
            ones += 1
        else:
            zeors += 1
        out["set"].append('train')
    else:
        hmm += 1
        
print(hmm, zeors, ones, hmm+zeors+ones)
out = pd.DataFrame(out)
out.to_csv("labelz.csv", index = False)
os.system("mv labelz.csv ../../data/gravo")
        
