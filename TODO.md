Objetivo:
 - Ter uma classe ClassicClassifier, ClinicClassifier cuja parent classe é TableClassifier
 - Ter uma classe RadiomicsLoader e TabularLoader cuja parent classe é TableLoader
 
ClinicClassifier apenas pode receber como argumento o TabularLoader

ClassicClassifier
 - fit
 - predict

ClinicClassifier
 - predict
 
TableClassifier
 - __init__(CSVLoader)
 - predict



CSVLoader
 - private split
 - private filter
 - private normalize
 - private set_sets
 - public  get_set(s)

    
    

RadiomicsLoader

TabularLoader
 - private impute
 - private ampute
 - private add_vars
 
RadiomicsMILLoader
