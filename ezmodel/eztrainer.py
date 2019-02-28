from sklearn.model_selection import train_test_split
from keras.models import Input,Model
from keras import layers

class eztrainer:

    def __init__(self):

        network = None
        X_train = None
        y_train = None
        X_valid = None
        y_valid = None

    def gen_trainval(self,ezdata,size=0.2,random_state=42):

        self.X_train,self.X_valid,self.y_train,self.y_valid = train_test_split(ezdata.X,ezdata.y,test_size=size,random_state=42)

        print ("[X] Train/Validation set generation (size = ",str(size),"): Done")
        print ("--- Training set : ", self.X_train.shape[0], "images")
        print ("--- Validation set     : ", self.X_valid.shape[0], "images")
        print("\n")

    def gen_network(self,inputs,outputs):
        m = Model(inputs=inputs,outputs=outputs)
        self.network = m

    def Input(self):
        return Input(shape=self.X_train.shape[1:])

    def ClassificationOutput(self,x0):

        if len(self.y_train.shape)==1:
            if len(np.unique(self.y_train))==2: #probably classification with binary class
                x = layers.Dense(1) (x0)
                x = layers.Activation("sigmoid") (x)
                return x
            else: #probably classification multiclass  or regression
                x = layers.Dense(1) (x0)
                x = layers.Activation("linear") (x)
                return x

        if len(self.y_train.shape)==2: #probably classification with multiclass
            x = layers.Dense(self.y_train.shape[1])(x0)
            x = layers.Activation("softmax") (x)
            return x

    def compile(self,optimizer):
            if hasattr(self,"network"):
                self.network.compile(**optimizer)
            else:
                print("[Fail] compile() : No network to compile the optimizer with. Please use gen_network() on your Keras network before.")
