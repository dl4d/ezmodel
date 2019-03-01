import sys, os
sys.path.append(os.path.abspath('..\\..\\ezmodel'))


from ezmodel.ezmodel   import ezmodel
from ezmodel.ezdata    import ezdata
from ezmodel.eztrainer import eztrainer



ez_data = ezdata()

print(ez_data.is_kernel())

ez_data.show_table(filename="C:\\Users\\daian\\Desktop\\DATA\\Iris\\iris.csv")
