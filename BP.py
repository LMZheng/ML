# ML
机器学习import numpy as np
from math import sqrt
#激活函数sigmoid
def sigmoid(x):
    y=(1/(1+np.exp(-x)))
    return y
#激活函数tanh
def tanh(x):
    y=(np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
    return y
#sigmoid激活函数的导数sigmoi_df
def sigmoid_df(y):
    dy=np.multiply((1-y),y)
    return dy
#tanh激活函数的导数
def tanh_df(y):
    dy=1-np.multiply(y,y)
    return dy

#不同激活函数对应的初试化区间
def inital_sigmoid(m,n):#sigmoid激活函数对应的初始化区间中随机抽取
    w1=np.mat(np.random.rand(m,n))
    w1=w1*(8.0*sqrt(6)/sqrt(m+n))-np.mat(np.ones((m,n)))*(4.0*sqrt(6)/sqrt(m+n))
    return w1
def inital_tanh(m,n):#tanh激活函数对应的初始化区间中随机抽取
    w=np.mat(np.random.rand(m,n))
    w1=w*(2.0*sqrt(6)/sqrt(m+n))-np.mat(np.ones((m,n)))*(sqrt(6)/sqrt(m+n))
    return w1

#以上为可选的引用函数，下面为BPNN算法内容
def predict(x,model):
    num=np.shape(x)[0]
    w1, w2= model['w1'], model['w2']
    #计算隐藏层
    x0=np.mat(np.ones((num,1)))
    x=np.hstack((x0,x))
    a1=x * w1
    a1=x * w1
    z1=tanh(a1)#此处选用了tanh激活函数,可以选用sigmoid
    z1=np.hstack((x0,z1))
    a2=z1*w2
    yhat=sigmoid(a2)
    return yhat

def bp_train(x,y,n_hidden,n_output,maxcyle,alpha,correct):
       
    '''计算隐含层的输入
    input:
    x:特征值
    y:目标值
    n_hidden：隐藏层节点数
    n_output：输出层节点数
    maxcyle:迭代次数
    alpha:学习率
    correct:动量因子，用于冲量项
    
    output:
    w1:输入层到隐藏层之间的权重
    w2:隐藏层到输出层之间的权重
    '''
    #初始化w1,w2
    num , n_input=np.shape(x)#num是样本个数，n_input是输入层的节点数
    x0=np.mat(np.ones((num,1)))
    x=np.hstack((x0,x))
    w1=inital_tanh(n_input+1,n_hidden)
    w2=inital_sigmoid(n_hidden+1,n_output)
    correction_w1=np.mat(np.random.rand(n_input+1,n_hidden))
    correction_w2=np.mat(np.random.rand(n_hidden+1,n_output))
    #训练 
    model = { 'w1': w1, 'w2': w2}
    i=0
    while i<=maxcyle:
        #正向传播
        a1=x * w1
        z1=tanh(a1)#此处隐藏层选用tanh激活函数,可以选用sigmoid
        z1=np.hstack((x0,z1))
        a2=z1*w2
        yhat=sigmoid(a2)
        
        #误差反向传播
        #输出层的残差
        q2=np.multiply((yhat-y),sigmoid_df(yhat))
        #隐藏层的残差
        q1=np.multiply((q2*w2.T),tanh_df(z1))
        #修正权重与偏置
        detla_w2=alpha*((z1.T)*q2)+correct*correction_w2
        detla_w1=alpha*((x.T)*q2)+correct*correction_w1
        w2=w2-detla_w2
        w1=w1-detla_w1
        model = { 'w1': w1,'w2': w2}
        correction_w2=detla_w2
        correction_w1=detla_w1
        i+=1                                                       
    return model
def bp_test(x_test, y_test, x_train, y_train, n_hidden=5, n_output=1, maxcyle=1000, alpha=0.01,correct=0.01):
    model=bp_train(x_train,y_train,n_hidden,n_output,maxcyle,alpha,correct)
    y_pred=predict(x_test,model)
    cost=np.multiply((y_pred-y_test),(y_pred-y_test))
    cost_sum=np.sum(cost)/np.shape(y_test)[0]
    #print("预测值：",y_pred)
    #print("损失函数：",cost_sum)
    return y_pred
#BPNN算法结束
    
    
#请输入一下参数
'''x_train:用于训练的特征数据集，要求是数组形式或者矩阵形式
    y_train：用于训练的目标值，要求同上
    x_test：测试特征值，要求同上
    y_test： 测试目标值，要求同上
    n_hidden:隐藏层的节点数，自行设定
    n_output:输出层节点数
    maxcyle:迭代次数
    alpha:学习率
    correct:动量因子'''
x_train=np.array([[0,0.1,2],[2,1,0],[3,41,1],[55,20,30],[90,78,50]])
y_train=np.array([[0],[0],[0],[1],[1]])
x_test=np.array([[0,3,7],[76,56,49]])
y_test=np.array([[0],[1]])
n_hidden=5
n_output=1
maxcyle=1000
alpha=0.01
correct=0.01

#利用BP算法计算预测值：隐藏层激活函数用的tanh,输出层转换函数用的sigmoid，在修正参数过程考虑了冲量项
y_pred=bp_test(x_test, y_test, x_train, y_train, n_hidden, n_output, maxcyle, alpha,correct)
