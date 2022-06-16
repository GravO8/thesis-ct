import pandas as pd

data = pd.read_csv("../../data/gravo/table_data.csv")
columns = ["dataNascimento-1", "gliceAd-4", "dataAVC-4", "totalNIHSS-5", "altVis-5", "altCons-5", "data-7", "rankin-23"]
data = data[["idProcessoLocal"]+columns].copy()

fake_strokes = [2523138, 111382, 515136, 779547, 2399328, 1391554, 1700004, 1810996, 1951046, 2369819, 2394862]
fake_strokes = [str(id_) for id_ in fake_strokes]
for id_ in fake_strokes:
    assert id_ not in data["idProcessoLocal"].values

missing = {"idProcessoLocal": data["idProcessoLocal"].values}
for col in columns:
    missing[col] = (data[col] == "None").astype(int)
missing["count"] = missing["dataNascimento-1"]+missing["gliceAd-4"]+missing["dataAVC-4"]+missing["totalNIHSS-5"]+missing["altVis-5"]+missing["altCons-5"]+missing["data-7"]+missing["rankin-23"]
missing = pd.DataFrame(missing)
missing = missing[missing["count"] != 0].copy()

for col in columns:
    missing[col] = ["missing" if r else None for r in missing[col].values]
    
missing = missing.sort_values(by = ["count"])
del missing["count"]
missing.rename(columns = {"dataNascimento-1":"dataNascimento", 
                "gliceAd-4":"gliceAd", 
                "dataAVC-4":"dataAVC", 
                "totalNIHSS-5":"totalNIHSS",
                "altVis-5":"altVis",
                "altCons-5":"altCons",
                "data-7":"data",
                "rankin-23":"rankin"}, inplace = True)
missing.to_csv("missing.csv", index = False, sep = ";")
