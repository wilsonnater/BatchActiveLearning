import numpy as np

class BatchActiveLearning:
    def __init__(
        self,
        BootStrapSize=0.8, #Fraction of strucutres in each bagging sample
        Num_Models=10, #Number of bagging model to make
        alpha=0.9,  #Weighting of Uncertainty(alpha) to Distance(1-alpha) 
        seed=None #Seed for random number generator
    ):

        self.BootStrapSize= BootStrapSize
        self.Num_Models=Num_Models
        self.alpha=alpha
        self.seed=seed 
        
        self.rng = np.random.default_rng(self.seed)
    
    def UncertaintyMetric(self,model,UnlabeledX,LabeledX,Y):
        AllPredicted=np.zeros([self.Num_Models,len(UnlabeledX)])
        for i in range(self.Num_Models):
            idx=np.arange(len(Y))
            self.rng.shuffle(idx)
            NumSamples=int(len(Y)*self.BootStrapSize)
            training_idx, test_idx = idx[:NumSamples], idx[NumSamples:]

            Train_X, Train_Y = LabeledX[training_idx], Y[training_idx]
            Test_X, Test_Y = LabeledX[test_idx], Y[test_idx]

            model.fit(Train_X,Train_Y)

            AllPredicted[i]=model.predict(UnlabeledX)
        return AllPredicted.std(axis=0)
    
    def Distance(self,LastSample,UnlabeledX):
        #This function is just to allow direct calling of distance through class
        return np.linalg.norm(UnlabeledX-LastSample,axis=1)
    
    def GetBatch(self,model,UnlabeledX,X,Y,BatchSize):
        NumSamples=int(len(Y)*self.BootStrapSize)
        NextBatch=[]
        
        AllSTD=self.UncertaintyMetric(model,UnlabeledX,X,Y)

        NextBatch.append(np.argmax(AllSTD))

        DistanceMetric=np.zeros([len(UnlabeledX)])
        for i in range(BatchSize-1):
            #Only needs to add the distance to the new selected point
            DistanceMetric+=self.Distance(UnlabeledX[NextBatch[-1]],UnlabeledX)

            Score=self.alpha*AllSTD+(1-self.alpha)*DistanceMetric

            NextBatch.append(np.argmax(Score))
        
        return NextBatch
