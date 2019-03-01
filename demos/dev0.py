import sys, os
sys.path.append(os.path.abspath('..\\..\\ezmodel'))


from ezmodel.ezmodel   import ezmodel
from ezmodel.ezdata    import ezdata
from ezmodel.eztrainer import eztrainer



ezdata().show_table(filename="C:\\Users\\daian\\Desktop\\DATA\\Iris\\iris.csv")

parameters={
    "name" : "Iris",
    "path" : "C:\\Users\\daian\\Desktop\\DATA\\Iris\\iris.csv",
    "table.target.column" : "species"
}

ez_data = ezdata(parameters)
