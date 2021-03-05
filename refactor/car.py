#导入需要的包


import os
os.environ["CUDA_VISIBLE_DEVICES"]='0'
import zipfile
import random
import json
import cv2
import numpy as np
from PIL import Image
import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph import Linear, Conv2D, BatchNorm
import matplotlib.pyplot as plt
from paddle.fluid.param_attr import ParamAttr

train_parameters = {
    "input_size": [1, 20, 20],                           #输入图片的shape
    "class_dim": -1,                                     #分类数
    "src_path":"data/data23617/characterData.zip",       #原始数据集路径
    "target_path":"/home/aistudio/data/dataset",        #要解压的路径 
    "train_list_path": "./train_data.txt",              #train_data.txt路径
    "eval_list_path": "./val_data.txt",                  #eval_data.txt路径
    "label_dict":{},                                    #标签字典
    "readme_path": "/home/aistudio/data/readme.json",   #readme.json路径
    "num_epochs": 1,                                    #训练轮数
    "train_batch_size": 32,                             #批次的大小
    "learning_strategy": {                              #优化函数相关的配置
        "lr": 0.001                                     #超参数学习率
    } 
}

def unzip_data(src_path,target_path):
    '''
    解压原始数据集，将src_path路径下的zip包解压至data/dataset目录下
    '''
    if(not os.path.isdir(target_path)):    
        z = zipfile.ZipFile(src_path, 'r')
        z.extractall(path=target_path)
        z.close()
    else:
        print("文件已解压")
        


def get_data_list(target_path,train_list_path,eval_list_path):
    '''
    生成数据列表
    '''
    #存放所有类别的信息
    class_detail = []
    #获取所有类别保存的文件夹名称
    data_list_path=target_path
    class_dirs = os.listdir(data_list_path)
    if '__MACOSX' in class_dirs:
        class_dirs.remove('__MACOSX')
    # #总的图像数量
    all_class_images = 0
    # #存放类别标签
    class_label=0
    # #存放类别数目
    class_dim = 0
    # #存储要写进eval.txt和train.txt中的内容
    trainer_list=[]
    eval_list=[]
    #读取每个类别
    for class_dir in class_dirs:
        if class_dir != ".DS_Store":
            class_dim += 1
            #每个类别的信息
            class_detail_list = {}
            eval_sum = 0
            trainer_sum = 0
            #统计每个类别有多少张图片
            class_sum = 0
            #获取类别路径 
            path = os.path.join(data_list_path,class_dir)
            # print(path)
            # 获取所有图片
            img_paths = os.listdir(path)
            for img_path in img_paths:                                  # 遍历文件夹下的每个图片
                if img_path =='.DS_Store':
                    continue
                name_path = os.path.join(path,img_path)                       # 每张图片的路径
                if class_sum % 10 == 0:                                 # 每10张图片取一个做验证数据
                    eval_sum += 1                                       # eval_sum为测试数据的数目
                    eval_list.append(name_path + "\t%d" % class_label + "\n")
                else:
                    trainer_sum += 1 
                    trainer_list.append(name_path + "\t%d" % class_label + "\n")#trainer_sum测试数据的数目
                class_sum += 1                                          #每类图片的数目
                all_class_images += 1                                   #所有类图片的数目
            
            # 说明的json文件的class_detail数据
            class_detail_list['class_name'] = class_dir             #类别名称
            class_detail_list['class_label'] = class_label          #类别标签
            class_detail_list['class_eval_images'] = eval_sum       #该类数据的测试集数目
            class_detail_list['class_trainer_images'] = trainer_sum #该类数据的训练集数目
            class_detail.append(class_detail_list)  
            #初始化标签列表
            train_parameters['label_dict'][str(class_label)] = class_dir
            class_label += 1
            
    #初始化分类数
    train_parameters['class_dim'] = class_dim
    print(train_parameters)
    #乱序  
    random.shuffle(eval_list)
    with open(eval_list_path, 'a', encoding='utf-8') as f:
        for eval_image in eval_list:
            f.write(eval_image) 
    #乱序        
    random.shuffle(trainer_list) 
    with open(train_list_path, 'a', encoding='utf-8') as f2:
        for train_image in trainer_list:
            f2.write(train_image) 

    # 说明的json文件信息
    readjson = {}
    readjson['all_class_name'] = data_list_path                  #文件父目录
    readjson['all_class_images'] = all_class_images
    readjson['class_detail'] = class_detail
    jsons = json.dumps(readjson, sort_keys=True, indent=4, separators=(',', ': '))
    with open(train_parameters['readme_path'],'w') as f:
        f.write(jsons)
    print ('生成数据列表完成！')


def data_reader(file_list):
    '''
    自定义data_reader
    '''
    def reader():
        with open(file_list, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f]
            for line in lines:
                img_path, lab = line.strip().split('\t')
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = np.array(img).astype('float32')
                img = img/255.0
                yield img, int(lab) 
    return reader


'''
参数初始化
'''
src_path=train_parameters['src_path']
target_path=train_parameters['target_path']
train_list_path=train_parameters['train_list_path']
eval_list_path=train_parameters['eval_list_path']
batch_size=train_parameters['train_batch_size']
'''
解压原始数据到指定路径
'''
unzip_data(src_path,target_path)

#每次生成数据列表前，首先清空train.txt和eval.txt
with open(train_list_path, 'w') as f: 
    f.seek(0)
    f.truncate() 
with open(eval_list_path, 'w') as f: 
    f.seek(0)
    f.truncate() 
    
#生成数据列表   
get_data_list(target_path,train_list_path,eval_list_path)

'''
构造数据提供器
'''
train_reader = paddle.batch(data_reader(train_list_path),
                            batch_size=batch_size,
                            drop_last=True)
eval_reader = paddle.batch(data_reader(eval_list_path),
                            batch_size=batch_size,
                            drop_last=True)


Batch=0
Batchs=[]
all_train_accs=[]
def draw_train_acc(Batchs, train_accs):
    title="training accs"
    plt.title(title, fontsize=24)
    plt.xlabel("batch", fontsize=14)
    plt.ylabel("acc", fontsize=14)
    plt.plot(Batchs, train_accs, color='green', label='training accs')
    plt.legend()
    plt.grid()
    plt.show()

all_train_loss=[]
def draw_train_loss(Batchs, train_loss):
    title="training loss"
    plt.title(title, fontsize=24)
    plt.xlabel("batch", fontsize=14)
    plt.ylabel("loss", fontsize=14)
    plt.plot(Batchs, train_loss, color='red', label='training loss')
    plt.legend()
    plt.grid()
    plt.show()

#定义DNN网络
class MyDNN(fluid.dygraph.Layer):
    '''
    DNN网络
    '''
    def __init__(self):
        super(MyDNN, self).__init__()
        self.conv1 = Conv2D(1, 64, 3, stride=1, padding=1, param_attr=ParamAttr(initializer=fluid.initializer.Normal(0.0, 0.02)), act="relu")
        self.bn1 = BatchNorm(64, 
                            param_attr=ParamAttr(
                                initializer=fluid.initializer.Normal(1., 0.02)
                            ),
                            bias_attr=ParamAttr(
                                initializer=fluid.initializer.Constant(0.0)
                            ))
        self.conv2 = Conv2D(64, 32, 3, stride=2, padding=1, 
                            param_attr=ParamAttr(
                                initializer=fluid.initializer.Normal(0.0, 0.02)
                            ),act="relu")
        self.bn2 = BatchNorm(32, 
                            param_attr=ParamAttr(
                                initializer=fluid.initializer.Normal(1., 0.02)
                            ),
                            bias_attr=ParamAttr(
                                initializer=fluid.initializer.Constant(0.0)
                            ))
        self.conv3 = Conv2D(32, 32, 3, stride=1, padding=1, 
                            param_attr=ParamAttr(
                                initializer=fluid.initializer.Normal(0.0, 0.02)
                            ),act="relu")
        self.bn3 = BatchNorm(32, 
                            param_attr=ParamAttr(
                                initializer=fluid.initializer.Normal(1., 0.02)
                            ),
                            bias_attr=ParamAttr(
                                initializer=fluid.initializer.Constant(0.0)
                            ))
        self.conv4 = Conv2D(32, 16, 3, stride=2, padding=1, 
                            param_attr=ParamAttr(
                                initializer=fluid.initializer.Normal(0.0, 0.02)
                            ),act="relu")
        self.bn4 = BatchNorm(16, 
                            param_attr=ParamAttr(
                                initializer=fluid.initializer.Normal(1., 0.02)
                            ),
                            bias_attr=ParamAttr(
                                initializer=fluid.initializer.Constant(0.0)
                            ))
        self.conv5 = Conv2D(64, 32, 3, stride=1, padding=1, 
                            param_attr=ParamAttr(
                                initializer=fluid.initializer.Normal(0.0, 0.02)
                            ),act="relu")
        self.ln1 = Linear(400 , 256, act='relu')
        self.ln2 = Linear(256, 64, act='softmax')
    def forward(self,input):        # forward 定义执行实际运行时网络的执行逻辑
        '''前向计算'''
        # x = fluid.layers.reshape(input, [input.shape[0], 1, 20, 20])
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x1 = self.conv3(x)
        # x = fluid.layers.concat(input=[x, x1], axis=1, name='concat')
        # x = self.conv5(x)
        x = self.bn3(x1)
        x = self.conv4(x)
        x = self.bn4(x)
        x = fluid.layers.reshape(x, [x.shape[0], 400])
        x = self.ln1(x)
        out = self.ln2(x)
        return out

with fluid.dygraph.guard():
    model=MyDNN() #模型实例化
    model.train() #训练模式
    opt=fluid.optimizer.SGDOptimizer(learning_rate=train_parameters['learning_strategy']['lr'], parameter_list=model.parameters())#优化器选用SGD随机梯度下降，学习率为0.001.
    epochs_num=train_parameters['num_epochs'] #迭代次数
    
    for pass_num in range(epochs_num):
        print(pass_num)
        for batch_id,data in enumerate(train_reader()):
            print(batch_id)
            images=np.array([x[0].reshape(1,20,20) for x in data],np.float32)
            labels = np.array([x[1] for x in data]).astype('int64')
            labels = labels[:, np.newaxis]
            image=fluid.dygraph.to_variable(images)
            label=fluid.dygraph.to_variable(labels)
            print("123")
            predict=model(image) #数据传入model
            print("123")
            print(predict.shape)
            print(label.shape)
            loss=fluid.layers.cross_entropy(predict,label)
            print("123")
            avg_loss=fluid.layers.mean(loss)#获取loss值
            print("123")
            acc=fluid.layers.accuracy(predict,label)#计算精度
            print("123")
            if batch_id!=0 and batch_id%50==0:
                Batch = Batch+50 
                Batchs.append(Batch)
                all_train_loss.append(avg_loss.numpy()[0])
                all_train_accs.append(acc.numpy()[0])
                
                print("train_pass:{},batch_id:{},train_loss:{},train_acc:{}".format(pass_num,batch_id,avg_loss.numpy(),acc.numpy()))
            
            avg_loss.backward()       
            opt.minimize(avg_loss)    #优化器对象的minimize方法对参数进行更新 
            model.clear_gradients()   #model.clear_gradients()来重置梯度
    fluid.save_dygraph(model.state_dict(),'MyDNN')#保存模型

draw_train_acc(Batchs,all_train_accs)
draw_train_loss(Batchs,all_train_loss)