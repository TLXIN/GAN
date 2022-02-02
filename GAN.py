#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 查看当前挂载的数据集目录, 该目录下的变更重启环境后会自动还原
# View dataset directory. 
# This directory will be recovered automatically after resetting environment. 
get_ipython().system('ls /home/aistudio/data')


# In[2]:


# 查看工作区文件, 该目录下的变更将会持久保存. 请及时清理不必要的文件, 避免加载过慢.
# View personal work directory. 
# All changes under this directory will be kept even after reset. 
# Please clean unnecessary files in time to speed up environment loading. 
get_ipython().system('ls /home/aistudio/work')


# In[3]:


# 如果需要进行持久化安装, 需要使用持久化路径, 如下方代码示例:
# If a persistence installation is required, 
# you need to use the persistence path as the following: 
get_ipython().system('mkdir /home/aistudio/external-libraries')
get_ipython().system('pip install beautifulsoup4 -t /home/aistudio/external-libraries')


# In[4]:


# 同时添加如下代码, 这样每次环境(kernel)启动的时候只要运行下方代码即可: 
# Also add the following code, 
# so that every time the environment (kernel) starts, 
# just run the following code: 
import sys 
sys.path.append('/home/aistudio/external-libraries')


# In[5]:


#导入必要的包
import numpy as np
import paddle
import paddle.fluid as pdflu
import matplotlib.pyplot as plt


# In[8]:


#噪声的维度
vector = 100
#vector = 50
#vector = 200
#这里因为数据集中的label没有用到，所以在此剔除
def data_reader(reader):
    def r():
        for img, label in reader():
            yield img.reshape(1, 28, 28)
    return r

#随机生成噪声，通过此来生成假图片

def vector_reader():

    while True:
        #将参数压缩为0.0-1.0
        yield np.random.normal(0.0, 1.0, (vector, 1, 1)).astype('float32')

# 生成真实图片reader
mnist_generator = paddle.batch(
    paddle.reader.shuffle(data_reader(paddle.dataset.mnist.train()), 30000),
    batch_size=128)

# 生成假图片的reader
z_generator = paddle.batch(vector_reader, batch_size=128)()
print("完成")


# In[9]:


#定义生成器
#由两组全连接层和BN层，两组转置卷积运算组成
def G(y, name="G"):
    #生成静态图需要参数唯一，在此使用pdflu.unique_name.guard使获得唯一参数
    with pdflu.unique_name.guard(name + "/"):
        #第一组全连接和BN层
        y = pdflu.layers.fc(y, size=1024, act='relu6')
        y = pdflu.layers.batch_norm(y, act='relu6')
        #第二组全连接和BN层
        y = pdflu.layers.fc(y, size=128 * 7 * 7)
        y = pdflu.layers.batch_norm(y, act='relu6')

        #进行形状变换
        #将vector转换为[128,7,7]形状的三维张量
        y = pdflu.layers.reshape(y, shape=(-1, 128, 7, 7))

        # 第一组转置卷积运算
        y = pdflu.layers.image_resize(y, scale=2)
        y = pdflu.layers.conv2d(y, num_filters=64, filter_size=5, padding=2, act='relu6')

        # 第二组转置卷积运算
        y = pdflu.layers.image_resize(y, scale=2)
        y = pdflu.layers.conv2d(y, num_filters=1, filter_size=5, padding=2, act='relu6') # 最后得到单通道灰度图

    return y


# In[10]:


#定义判别器
#使用三个卷积层和一个全连接层构成
def D(images, name="D"):
    #卷积层和BN层
    def conv_bn(input, num_filters, filter_size):
        y = pdflu.layers.conv2d(input=input,
                                num_filters=num_filters,
                                filter_size=filter_size,
                                stride=1,
                                bias_attr=False)
        y = pdflu.layers.batch_norm(y, act="leaky_relu")
        return y


    with pdflu.unique_name.guard(name + "/"):
        #第一组卷积池化
        y = conv_bn(images, num_filters=64, filter_size=3)
        y = pdflu.layers.pool2d(y, pool_size=2, pool_stride=2)
        #第二组卷积池化
        y = conv_bn(y, num_filters=128, filter_size=3)
        y = pdflu.layers.pool2d(y, pool_size=2, pool_stride=2)
        
        #全连接输出层
        y = pdflu.layers.fc(input=y, size=1024)
        y = pdflu.layers.batch_norm(input=y, act='leaky_relu')
        y = pdflu.layers.fc(y, size=1)
        #输出即为评分，即概率

    return y


# In[11]:


#创建四个program用于分别进行训练生成器生成图片，训练判别器识别真实图片，训练判别器识别假图片和初始化参数
#1.判别器判断假图片
train_d_fake = pdflu.Program()

#2.判别器判断真图片
train_d_real = pdflu.Program()

#3.生成器生成图片
train_g = pdflu.Program()

#4.初始化
startup = pdflu.Program()


# In[12]:


# 从Program获取prefix开头的参数名字
# 取出模型参数
def get_params(program, prefix):

    all_params = program.global_block().all_parameters()

    return [t.name for t in all_params if t.name.startswith(prefix)]



# In[13]:


#1.判别器识别假图片
import paddle
paddle.enable_static()
with pdflu.program_guard(train_d_fake, startup):
    z = pdflu.data(name='z', shape=[None,vector, 1, 1])
    zeros = pdflu.layers.fill_constant_batch_size_like(z, shape=[-1, 1], dtype='float32', value=0)#创建一个张量，用value中提供的常量来初始化张量

    #判断假图片的概率
    p_fake = D(G(z))

    #获取损失函数
    #这里即让判别器对于假图片的判断接近于0
    fake_cost = pdflu.layers.sigmoid_cross_entropy_with_logits(p_fake, zeros)
    #平均
    fake_avg_cost = pdflu.layers.mean(fake_cost)

    #获取判别器的参数
    d_params = get_params(train_d_fake, "D")


    #优化
    optimizer = pdflu.optimizer.AdamOptimizer(learning_rate=2e-4)
    optimizer.minimize(fake_avg_cost, parameter_list=d_params)




# In[14]:


#2.判别器识别真图片

with pdflu.program_guard(train_d_real, startup):
    real_image = pdflu.data('image', shape=[None,1, 28, 28])
    ones = pdflu.layers.fill_constant_batch_size_like(real_image, shape=[-1, 1], dtype='float32', value=1)

    #判断真图片的概率
    p_real = D(real_image)

    #获取损失函数
    #让判别器对于真图片的判断接近为1
    real_cost = pdflu.layers.sigmoid_cross_entropy_with_logits(p_real, ones)
    real_avg_cost = pdflu.layers.mean(real_cost)


    #获取判别器的参数
    d_params = get_params(train_d_real, "D")

    #优化
    optimizer = pdflu.optimizer.AdamOptimizer(learning_rate=2e-4)
    optimizer.minimize(real_avg_cost, parameter_list=d_params)


# In[15]:


#3.生成器生成假图片

with pdflu.program_guard(train_g, startup):
    z = pdflu.data(name='z', shape=[None,vector, 1, 1])
    ones = pdflu.layers.fill_constant_batch_size_like(z, shape=[-1, 1], dtype='float32', value=1)

    #生成图片
    fake = G(z)

    #克隆预测程序，用于训练生成器生成评分高的图片
    infer_program = train_g.clone(for_test=True)

    #生成符合判别器的假图片
    p = D(fake)

    # 获取损失函数
    g_cost = pdflu.layers.sigmoid_cross_entropy_with_logits(p, ones)
    g_avg_cost = pdflu.layers.mean(g_cost)

    #获取G的参数
    g_params = get_params(train_g, "G")

    #优化参数，固定D，训练G
    optimizer = pdflu.optimizer.AdamOptimizer(learning_rate=2e-4)
    optimizer.minimize(g_avg_cost, parameter_list=g_params)



# In[16]:


#可视化

def show_image_grid(images, pass_id=None):
    fig = plt.figure(figsize=(5, 5))
    fig.suptitle("Pass {}".format(pass_id))
    gs = plt.GridSpec(8, 8)
    gs.update(wspace=0.05, hspace=0.05)


    for i, image in enumerate(images[:64]):

        ax = plt.subplot(gs[i])

        plt.axis('off')

        ax.set_xticklabels([])

        ax.set_yticklabels([])

        ax.set_aspect('equal')

        plt.imshow(image[0], cmap='Greys_r')

    plt.show()

def draw_d_train_process(title,iters,fake_costs,real_costs,label_r_fake,label_r_real):
    plt.title(title, fontsize=24)
    plt.xlabel("iter", fontsize=20)
    plt.ylabel("cost", fontsize=20)
    
    plt.plot(iters, fake_costs,color='red',label=label_r_fake) 
    plt.plot(iters, real_costs,color='green',label=label_r_real)  

    plt.legend()
    plt.grid()
    plt.show()

def draw_g_train_process(title,iters,g_costs,label_r_g):
    plt.title(title, fontsize=24)
    plt.xlabel("iter", fontsize=20)
    plt.ylabel("cost", fontsize=20)
    
    plt.plot(iters, g_costs,color='red',label=label_r_g)  

    plt.legend()
    plt.grid()
    plt.show()


# In[19]:


# 创建解析器

#取决是否使用GPU还是CPU
use_cuda = True
place = pdflu.CUDAPlace(0) if use_cuda else pdflu.CPUPlace()
exe = pdflu.Executor(place)

#初始化参数
exe.run(startup)
my_iter = 0


# In[18]:


#测试
import numpy as np
test_z = np.array(next(z_generator))


# In[ ]:



#开始训练

my_r_fake =[]
my_r_real = []
my_r_g = []
my_iters =[]

for pass_id in range(50):
    my_iter += 1
    for i, real_image in enumerate(mnist_generator()):

        
        # 训练判别器D识别生成器G生成的假图片
        
        r_fake = exe.run(program=train_d_fake,

                         fetch_list=[fake_avg_cost],

                         feed={'z': np.array(next(z_generator))})


        # 训练判别器D识别真实图片

        r_real = exe.run(program=train_d_real,

                         fetch_list=[real_avg_cost],

                         feed={'image': np.array(real_image)})


        # 训练生成器G生成符合判别器D标准的假图片

        r_g = exe.run(program=train_g,

                      fetch_list=[g_avg_cost],

                      feed={'z': np.array(next(z_generator))})

    print("Pass：%d，fake_avg_cost：%f, real_avg_cost：%f, g_avg_cost：%f" % (pass_id, r_fake[0][0], r_real[0][0], r_g[0][0]))
    print(r_fake)
    my_iters.append(my_iter)
    my_r_fake.append(r_fake[0][0])
    my_r_real.append(r_real[0][0])
    my_r_g.append(r_g[0][0])

    # 测试生成的图片

    r_i = exe.run(program=infer_program,

                  fetch_list=[fake],

                  feed={'z': test_z})


    # 显示生成的图片

    show_image_grid(r_i[0], pass_id)
    
draw_d_train_process("training_D",my_iters,my_r_fake,my_r_real,"fake_avg_cost","real_avg_cost")
draw_g_train_process("training_G",my_iters,my_r_g,"g_avg_cost")


# In[ ]:





# 
