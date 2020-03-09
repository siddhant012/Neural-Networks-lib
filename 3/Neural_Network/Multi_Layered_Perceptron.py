import numpy as np
import math

class perceptron:


    def __init__(self):
        self.num_of_neurons=[2,3,1]
        self.num_of_layers=2
        self.num_of_features=self.num_of_neurons[0]
        self.acti_types=[1,1,1,1]
        self.num_of_iterations=100
        self.learning_rate=2.0
        self.num_of_training_examples=4
        self.num_of_test_examples = 0
        self.training_inputdata=[]
        self.test_input_data=[]
        self.training_output_data = []
        self.test_output_data=[]
        self.a=[]
        self.z=[]
        self.w = []
        self.b=[]
        self.dw_data=[]   # used in case of gradient checking
        self.db_data=[]   # used in case of gradient checking
        self.rmse=0.0
        self.training_input_data=(np.zeros([self.num_of_training_examples,self.num_of_features]))
        self.training_output_data=(np.zeros([self.num_of_training_examples,self.num_of_neurons[self.num_of_layers]]))
        for i in range(self.num_of_layers): self.w.append(np.random.rand(self.num_of_neurons[i+1],self.num_of_neurons[i]))
        for i in range(self.num_of_layers+1): self.z.append(np.zeros([self.num_of_neurons[i],self.num_of_training_examples]))
        for i in range(self.num_of_layers+1): self.a.append(np.zeros([self.num_of_neurons[i],self.num_of_training_examples]))
        for i in range(self.num_of_layers): self.b.append(np.zeros([self.num_of_neurons[i+1],1]))


    def __acti(self,acti_type,num):
        if(acti_type==1): return 1/(1+np.exp(-num))
        elif(acti_type==2):pass
        elif(acti_type==3):pass
        else:pass

    def __acti_der(self,acti_type,num):
        if(acti_type==1):
            t=1/(1+np.exp(-num))
            return t*(1-t)
        elif(acti_type==2):pass
        elif(acti_type==3):pass
        else:pass

    def __cost_der(self,acti_type,predicted,real):
        if(acti_type==1):
            temp=[]
            for i in range(0,self.num_of_training_examples):
                sml=1.0e-30
                temp.append((-real[0][i]/(predicted[0][i]+sml)+(1-real[0][i])/(1-predicted[0][i]+sml))/self.num_of_training_examples)
            return np.array(temp).reshape(self.num_of_training_examples,1)



    def load_network(self):
        self.w=[]
        self.z=[]
        self.a=[]
        self.b=[]
        for i in range(self.num_of_layers): self.w.append(np.random.rand(self.num_of_neurons[i+1],self.num_of_neurons[i]))
        for i in range(self.num_of_layers+1): self.z.append(np.zeros([self.num_of_neurons[i],self.num_of_training_examples]))
        for i in range(self.num_of_layers+1): self.a.append(np.zeros([self.num_of_neurons[i],self.num_of_training_examples]))
        for i in range(self.num_of_layers): self.b.append(np.zeros([self.num_of_neurons[i+1],1]))


    def calculate_rmse(self,input_data,output_data,num_of_test_cases):
        self.predict(input_data)
        self.rmse=np.multiply( (output_data-self.a[self.num_of_layers].transpose()) , (output_data-self.a[self.num_of_layers].transpose()) ).sum()
        self.rmse/=num_of_test_cases
        return self.rmse


    def calculate_cost(self,input_data,output_data,num_of_test_cases):
        self.predict(self.training_input_data)
        cost=0.0
        for i in range(num_of_test_cases):
            for j in range(self.num_of_neurons[self.num_of_layers]):
                cost+= output_data[i][j]*math.log(self.a[self.num_of_layers][j][i],math.exp(1))
                cost+= (1-output_data[i][j])*(1-math.log(self.a[self.num_of_layers][j][i],math.exp(1)))
        cost/=num_of_test_cases
        print("cost:",cost)
        return cost
        



    def train(self):

        print("training...")

        for i in range(self.num_of_iterations):
             self.dw_data=[]  #used for gradient checking
             self.db_data=[]  #used for gradient checking

             #forward pass
             self.a[0] = (np.array(self.training_input_data)).transpose()
             for j in range(1,self.num_of_layers+1):
                 self.z[j]=np.dot(self.w[j-1],self.a[j-1])+self.b[j-1]
                 self.a[j]=self.__acti(self.acti_types[j],self.z[j])

             #backward pass
             da = self.__cost_der(self.acti_types[self.num_of_layers],self.a[self.num_of_layers],np.array(self.training_output_data).transpose()).transpose()
             for j in range(self.num_of_layers,0,-1):
                 dz=np.multiply(da,self.__acti_der(self.acti_types[j],self.z[j]))
                 da=np.dot(self.w[j-1].transpose(),dz)

                 dw=np.dot(dz,self.a[j-1].transpose())*(1/self.num_of_training_examples)
                 db=dz.sum()*(1/self.num_of_training_examples)


                 self.dw_data.append(dw)  #used for gradient checking
                 self.db_data.append(db*np.ones((self.num_of_neurons[j],1)))  #used for gradient checking


                 self.w[j-1]=self.w[j-1]-self.learning_rate*dw
                 self.b[j-1]=self.b[j-1]-self.learning_rate*db
             temp=self.rmse
             self.calculate_rmse(self.training_input_data,self.training_output_data,self.num_of_training_examples)
             #if(self.rmse-temp<=-1.0e-6): self.learning_rate*=1.1
             #if(self.rmse-temp>= 1.0e-6): self.learning_rate*=0.9
             print('iteration '+str(i)+' complete' , 'rmse=',self.rmse,'    ', '('+str(self.rmse-temp)+')' , '      learning rate=',self.learning_rate)

        self.calculate_rmse(self.training_input_data,self.training_output_data,self.num_of_training_examples)
        print("Training data RMSE:",self.rmse)
        if(self.test_input_data):
            self.calculate_rmse(self.test_input_data,self.test_output_data,self.num_of_test_examples)




    def predict(self,prediction_data):
        self.a[0] = (np.array(prediction_data)).transpose()
        for j in range(1,self.num_of_layers + 1):
            self.z[j] = np.dot(self.w[j - 1],self.a[j - 1]) + self.b[j - 1]
            self.a[j] = self.__acti(self.acti_types[j],self.z[j])
        return self.a[self.num_of_layers]

    def __randomize(self):
        pass

def main():
    p1=perceptron()
    p1.training_input_data = [[1,1],[0,0],[0,1],[1,0]]
    p1.training_output_data=[[1],[1],[0],[0]]
    p1.no_of_training_examples=4
    p1.num_of_iterations=1000
    p1.learning_rate=5
    p1.load_network()
    p1.train()
    print(p1.a[2])
    p1.predict([ [0.1,-0.1] , [1.2,0.97] ,[1.12,0.21] , [0.17,1.11] ])

    # gradient checking implementation
    thet=[]
    d_thet=[]
    epsilon=1.0e-7
    p1.dw_data=p1.dw_data[::-1]
    p1.db_data=p1.db_data[::-1]
    
    for i in range(0,p1.num_of_layers):
        thet.append(p1.w[i].flatten().tolist())
        thet.append(p1.b[i].flatten().tolist())

        d_thet.append(p1.dw_data[i].flatten().tolist())
        d_thet.append(p1.db_data[i].flatten().tolist())
    
    theta=[]
    d_theta=[]
    for i in thet:
        for j in i:theta.append(j)
    for i in d_thet:
        for j in i:d_theta.append(j) 
    #print(theta)
    #print(d_theta)
    
    counter=0
    error=0.0
    error_arr=[]
    for i in range(p1.num_of_layers):

        for j in range(p1.num_of_neurons[i+1]):
            for k in range(p1.num_of_neurons[i]):
                counter+=1

                temp=p1.w[i][j][k]
                p1.w[i][j][k]+=epsilon
                cost1=p1.calculate_cost(p1.training_input_data,p1.training_output_data,p1.num_of_training_examples)

                p1.w[i][j][k]=temp
                p1.w[i][j][k]-=epsilon
                cost2=p1.calculate_cost(p1.training_input_data,p1.training_output_data,p1.num_of_training_examples)

                p1.w[i][j][k]=temp

                der=(cost1-cost2)/(2*epsilon)
                error_arr.append(der)


        for j in range(p1.num_of_neurons[i+1]):
            counter+=1

            temp=p1.b[i][j]
            p1.b[i][j]+=epsilon
            cost1=p1.calculate_cost(p1.training_input_data,p1.training_output_data,p1.num_of_training_examples)

            p1.b[i][j]=temp
            p1.b[i][j]-=epsilon
            cost2=p1.calculate_cost(p1.training_input_data,p1.training_output_data,p1.num_of_training_examples)

            p1.b[i][j]=temp
            
            der=(cost1-cost2)/(2*epsilon)
            error_arr.append(der)

    error=0.0
    t1=0
    t2=0
    for i in range(len(error_arr)):
        error+= (d_theta[i]-error_arr[i])**2
        t1+= d_theta[i]*d_theta[i]
        t2+= error_arr[i]*error_arr[i]
    error=error**0.5
    error/= (t1**0.5 + t2**0.5)

    print("ERROR IS:",error)
    print(error_arr)
    print(d_theta)







        






    





if __name__ == '__main__':
    main()