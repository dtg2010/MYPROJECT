#!/usr/bin/env python
# coding: utf-8

# In[1]:



import pandas as pd
import numpy as np
import paddle
import paddle.fluid as fluid
import matplotlib.pyplot as plt
from paddle.utils.plot import Ploter
import os

get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


global data,train_data,test_data,maximums,minimums,averages

#train_data_path = "data/data1627/train.csv" # 训练数据集
#test_data_path = "data/data1627/test.csv" # 待预测数据


BATCH_SIZE = 10 # BATCH_SIZE = ?
LEARNING_RATE = 0.01 # LEARNING_RATE = ?
TRAIN_TEST_RATIO = 0.6 # 训练数据占训练数据集的比例
attribute = 'myattribute' # 待预测的属性


# In[5]:


#
SAVE_DIRNAME = 'model'
# 获得当前文件夹
cur_dir = os.path.dirname(os.path.realpath("__file__"))
# 获得文件路径
filename = cur_dir + "/data/mydata.txt"
f = open(filename) 
df = f.readlines()    
f.close()


# In[6]:


data = []
for line in df:
    data_raw = line.strip('\n').strip('\r').split('\t') #这里data_raw是列表形式，代表一行数据样本
    data.append(data_raw)#data为二维列表形式
data = np.array(data, dtype='float32')


# In[7]:


print('数据类型：',type(data))
print('数据个数：', data.shape[0])
print('数据形状：', data.shape)
print('数据第一行：', data[0])
print(data)


# In[8]:


def normalization(data):
    global maximums,minimums,averages
    maximums, minimums, averages = data.max(axis = 0), data.min(axis = 0), data.sum(axis = 0)/data.shape[0]
    feature_number = 10 #未对Label归一化
    for index in range(feature_number-1):
        #练习开始，约1行代码
        data[:, index] = (data[:, index] - averages[index] ) / (maximums[index] - minimums[index])
        #结束
    return data

data = normalization(data)


# In[9]:


data


# In[10]:


train_data = data[:int(data.shape[0] * TRAIN_TEST_RATIO)].copy()#训练数据
test_data = data[int(data.shape[0] * TRAIN_TEST_RATIO):].copy()#测试数据


# In[11]:



data.shape, train_data.shape, test_data.shape


# In[12]:



#使用yield关键字返回generator以提高效率
def read_data(data):
    def reader():
        for row in data:
            #练习开始，1行代码，使用yield关键字
            yield np.array(row[:8]), np.array(row[8])
            #结束
    return reader
#给定数据作为参数，返回生成器

#定义用于训练过程reader的generator
def train():
    global train_data
    return read_data(train_data)
#定义用于测试过程reader的generator
def test():
    global test_data
    return read_data(test_data)


# In[13]:


train_reader = paddle.batch(paddle.reader.shuffle(train(), buf_size=500), batch_size=BATCH_SIZE)
test_reader = paddle.batch(paddle.reader.shuffle(test(), buf_size=500), batch_size=BATCH_SIZE)


# In[14]:


x=fluid.layers.data(name='x',shape=[8],dtype='float32')
y=fluid.layers.data(name='y',shape=[1],dtype='float32')
y_predict = fluid.layers.fc(input=x, size=1, act=None)


# In[15]:


cost = fluid.layers.square_error_cost(input=y_predict, label=y) # 求一个batch的损失值
avg_cost = fluid.layers.mean(cost) # 对损失值求平均值
acc=fluid.layers.accuracy(input=y_predict,label=y)


# In[16]:


test_program = fluid.default_main_program().clone(for_test=True)


# In[17]:


optimizer = fluid.optimizer.SGDOptimizer(learning_rate=LEARNING_RATE)
opts = optimizer.minimize(avg_cost)


# In[18]:


def event_handler(pass_id, batch_id, cost):
    # 打印训练的中间结果，训练轮次，batch数，损失函数
    print("Pass %d, Batch %d, Cost %f" % (pass_id, batch_id, cost))


# In[19]:



from paddle.utils.plot import Ploter

train_prompt = "Train cost"
test_prompt = "Test cost"
cost_ploter = Ploter(train_prompt, test_prompt)

# 将训练过程绘图表示
def event_handler_plot(ploter_title, step, cost):
    cost_ploter.append(ploter_title, step, cost)
    #cost_ploter.plot()


# In[20]:


use_cuda = False # 如想使用GPU，请设置为 True
place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()


# In[21]:


def train_test(train_test_program,
                   train_test_feed, train_test_reader):
    # 将平均损失存储在avg_loss_set中
    avg_loss_set = []
    # 将测试 reader yield 出的每一个数据传入网络中进行训练
    for test_data in train_test_reader():
        avg_loss_np = exe.run(
            program=train_test_program,
            feed=train_test_feed.feed(test_data),
            fetch_list=[loss])
        avg_loss_set.append(float(avg_loss_np))
    # 获得测试数据上的准确率和损失值
    avg_loss_val_mean = np.array(avg_loss_set).mean()
    # 返回平均损失值，平均准确率
    return avg_loss_val_mean


# In[22]:


exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())


# In[23]:


feeder = fluid.DataFeeder(place=place, feed_list=[x, y])


# In[24]:



iter=0;
iters=[]
train_costs=[]

def draw_train_process(iters,train_costs):
    title="training cost"
    plt.title(title, fontsize=24)
    plt.xlabel("iter", fontsize=14)
    plt.ylabel("cost", fontsize=14)
    plt.plot(iters, train_costs,color='red',label='training cost') 
    plt.grid()
    plt.show()


# In[25]:


EPOCH_NUM=50
model_save_dir = "/home/aistudio/work/fit_a_line.inference.model"

for pass_id in range(EPOCH_NUM):                                  #训练EPOCH_NUM轮
    # 开始训练并输出最后一个batch的损失值
    train_cost = 0
    for batch_id, data in enumerate(train_reader()):              #遍历train_reader迭代器
        train_cost= exe.run(program=fluid.default_main_program(),#运行主程序
                             feed=feeder.feed(data),              #喂入一个batch的训练数据，根据feed_list和data提供的信息，将输入数据转成一种特殊的数据结构
                             fetch_list=[avg_cost])    
        if batch_id % 40 == 0:
            print("Pass:%d, Cost:%0.5f" % (pass_id, train_cost[0][0]))    #打印最后一个batch的损失值
        iter=iter+BATCH_SIZE
        iters.append(iter)
        train_costs.append(train_cost[0][0])
       
   
    # 开始测试并输出最后一个batch的损失值
    test_cost = 0
    for batch_id, data in enumerate(test_reader()):               #遍历test_reader迭代器
        test_cost= exe.run(program=test_program, #运行测试cheng
                            feed=feeder.feed(data),               #喂入一个batch的测试数据
                            fetch_list=[avg_cost])                #fetch均方误差
    print('Test:%d, Cost:%0.5f' % (pass_id, test_cost[0][0]))     #打印最后一个batch的损失值
    
    #保存模型
    # 如果保存路径不存在就创建
if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)
print ('save models to %s' % (model_save_dir))
#保存训练参数到指定路径中，构建一个专门用预测的program
fluid.io.save_inference_model(model_save_dir,   #保存推理model的路径
                                  ['x'],            #推理（inference）需要 feed 的数据
                                  [y_predict],      #保存推理（inference）结果的 Variables
                                  exe)              #exe 保存 inference model
draw_train_process(iters,train_costs)


# In[ ]:


infer_exe = fluid.Executor(place)    #创建推测用的executor
inference_scope = fluid.core.Scope() #Scope指定作用域


# In[ ]:


infer_results=[]
groud_truths=[]

#绘制真实值和预测值对比图
def draw_infer_result(groud_truths,infer_results):
    title='myattribute '
    plt.title(title, fontsize=24)
    x = np.arange(0,1) 
    y = x
    plt.plot(x, y)
    plt.xlabel('ground truth', fontsize=14)
    plt.ylabel('infer result', fontsize=14)
    plt.scatter(groud_truths, infer_results,color='green',label='training cost') 
    plt.grid()
    plt.show()


# In[ ]:


with fluid.scope_guard(inference_scope):#修改全局/默认作用域（scope）, 运行时中的所有变量都将分配给新的scope。
    #从指定目录中加载 推理model(inference model)
    [inference_program,                             #推理的program
     feed_target_names,                             #需要在推理program中提供数据的变量名称
     fetch_targets] = fluid.io.load_inference_model(#fetch_targets: 推断结果
                                    model_save_dir, #model_save_dir:模型训练路径 
                                    infer_exe)      #infer_exe: 预测用executor
    #获取预测数据
    infer_reader = paddle.batch(paddle.reader.shuffle(test(), buf_size=500), batch_size=BATCH_SIZE)
    #从test_reader中分割x
    test_data = next(infer_reader())
    test_x = np.array([data[0] for data in test_data]).astype("float32")
    test_y= np.array([data[1] for data in test_data]).astype("float32")
    results = infer_exe.run(inference_program,                              #预测模型
                            feed={feed_target_names[0]: np.array(test_x)},  #喂入要预测的x值
                            fetch_list=fetch_targets)                       #得到推测结果 
                            
    print("infer results: (myattribute)")
    for idx, val in enumerate(results[0]):
        print("%d: %.2f" % (idx, val))
        infer_results.append(val)
    print("ground truth:")
    for idx, val in enumerate(test_y):
        print("%d: %.2f" % (idx, val))
        groud_truths.append(val)
    draw_infer_result(groud_truths,infer_results)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




