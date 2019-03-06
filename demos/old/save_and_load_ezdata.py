import sys, os
sys.path.append(os.path.abspath('..\\..\\ezmodel'))

from ezmodel.ezmodel   import ezmodel
from ezmodel.ezdata    import ezdata
from ezmodel.eztrainer import eztrainer
# --------------------  [EZ Model Initialization]  ---------------------------

ez_model = ezmodel(type="classification")

# ----------------------------  [EZ Data]  -----------------------------------
# Table Classification from a csv file:
# - One subdirectory by Class
parameters = {
    "name"                      : "Iris",
    "type"                      : "classification",
    "format"                    : "table",
    "from"                      : "file",
    "path"                      : "C:\\Users\\daian\\Desktop\\DATA\\Iris\\iris.csv",
    "table_delimiter"           : ",",
    "table_target_column"       : ["species"],
    "table_target_column_type"  : "string"
}

ez_data = ezdata(parameters)

ez_data.gen_test(size=0.2)

ez_data.preprocess(X="standard",y="categorical") #on crée les scaler dans ez_data aussi

# ----------------------------  [EZ Data Save]  -------------------------------

ez_data.save("iris_dataset")

# ----------------------------  [EZ Data Load]  -------------------------------

ez_data2 = ezdata(load="iris_dataset")

print(ez_data2.params["name"])
