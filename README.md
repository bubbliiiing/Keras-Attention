## Attention：注意力机制在Keras当中的实现
---

## 目录
1. [所需环境 Environment](#所需环境)
2. [LSTM中的注意力机制](#LSTM中的注意力机制)
3. [Conv中的注意力机制](#Conv中的注意力机制)

## 所需环境
tensorflow-gpu==1.13.1  
keras==2.1.5  

## LSTM中的注意力机制
在本库中，我将注意力机制施加在LSTM的Step上，目的是注意输入进来的样本，每一个Step的重要程度。我们使用的样本数据如下：
```python
X = [[-21.03816538   1.4249185 ]
     [  3.76040424 -12.83660875]
     [  1.           1.        ]
     [-10.17242648   5.37333323]
     [  2.97058584  -9.31965078]
     [  3.69295417   8.47650258]
     [ -6.91492102  11.00583167]
     [ -0.03511656  -1.71475966]
     [ 10.9554255   12.47562052]
     [ -5.70470182   4.70055424]]
Y = [1]
```
我们可以看到，当我们将attention_column设置为2的时候，第2个step的输入和当前batch的输出相同，其它step的值是随机设定的，因此网络应该需要去注意第2个step的输入，这就是我们希望他注意的情况。

## Conv中的注意力机制
在卷积神经网络中，我将注意力机制施加在通道上，即，注意输入进来的特征层每一个通道的比重。利用该注意力机制，可以获得每个通道的重要程度。如下：
```python
#---------------------------------------#
#   通道注意力机制单元
#   利用两次全连接算出每个通道的比重
#   可以连接在任意特征层后面
#---------------------------------------#
def squeeze(inputs):
    input_channels = int(inputs.shape[-1])
    x = GlobalAveragePooling2D()(inputs)

    x = Dense(int(input_channels/4))(x)
    x = Activation(relu6)(x)

    x = Dense(input_channels)(x)
    x = Activation(hard_swish)(x)

    x = Reshape((1, 1, input_channels))(x)
    x = Multiply()([inputs, x])
    return x
```
