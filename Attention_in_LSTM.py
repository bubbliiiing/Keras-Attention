import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.layers import merge
from keras.layers.core import *
from keras.layers.recurrent import LSTM
from keras.models import *

'''
获得数据集、attention_column代表我们希望被注意的列
这个数据集是我们人为创建的，目的是为了演示注意力机制，示例如下：
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
我们可以看到，当我们将attention_column设置为2的时候
第2个step的输入和当前batch的输出相同，其它step的值是随机设定的
因此网络应该需要去注意第2个step的输入，这就是我们希望他注意的情况。
'''
def get_data_recurrent(n, time_steps, input_dim, attention_column=2):
    x = np.random.normal(loc=0, scale=10, size=(n, time_steps, input_dim))
    y = np.random.randint(low=0, high=2, size=(n, 1))
    x[:, attention_column, :] = np.tile(y[:], (1, input_dim))
    return x, y

#-------------------------------------------#
#   对每一个step的注意力权值
#-------------------------------------------#
def get_activations(model, inputs, layer_name=None):
    inp = model.input
    for layer in model.layers:
        if layer.name == layer_name:
            Y = layer.output
    model = Model(inp,Y)
    out = model.predict(inputs)
    out = np.mean(out[0],axis=-1)
    return out
    
#------------------------------------------------------------------------------------------------------#
#   注意力模块，主要是实现对step维度的注意力机制
#   在这里大家可能会疑惑，为什么需要先Permute再进行注意力机制的施加。
#   这是因为，如果我们直接进行全连接的话，我们的最后一维是特征维度，这个时候，我们每个step的特征是分开的，
#   此时进行全连接的话，得出来注意力权值每一个step之间是不存在特征交换的，自然也就不准确了。
#   所以在这里我们需要首先将step维度转到最后一维，然后再进行全连接，根据每一个step的特征获得注意力机制的权值。
#------------------------------------------------------------------------------------------------------#
def attention_3d_block(inputs):
    # batch_size, time_steps, lstm_units -> batch_size, lstm_units, time_steps
    a = Permute((2, 1))(inputs)
    # batch_size, lstm_units, time_steps -> batch_size, lstm_units, time_steps
    a = Dense(TIME_STEPS, activation='softmax')(a)
    # batch_size, lstm_units, time_steps -> batch_size, time_steps, lstm_units
    a_probs = Permute((2, 1), name='attention_vec')(a)

    # 相当于获得每一个step中，每个特征的权重
    output_attention_mul = merge([inputs, a_probs], name='attention_mul', mode='mul')
    return output_attention_mul

#-------------------------------------------#
#  建立注意力模型
#-------------------------------------------#
def get_attention_model(time_steps, input_dim, lstm_units = 32):
    inputs = Input(shape=(time_steps, input_dim,))
    
    # (batch_size, time_steps, input_dim) -> (batch_size, input_dim, lstm_units)
    lstm_out = LSTM(lstm_units, return_sequences=True)(inputs)

    attention_mul = attention_3d_block(lstm_out)
    # (batch_size, input_dim, lstm_units) -> (batch_size, input_dim*lstm_units)
    attention_mul = Flatten()(attention_mul)
    output = Dense(1, activation='sigmoid')(attention_mul)
    model = Model(input=[inputs], output=output)
    return model

if __name__ == '__main__':
    N = 100000
    INPUT_DIM = 2
    TIME_STEPS = 10
    #------------------------------------------------------#
    #   每一个输入样本的step为10，每一个step的数据长度为2
    #   X - batch, 10, 2
    #   Y - batch, 1
    #------------------------------------------------------#
    X, Y = get_data_recurrent(N, TIME_STEPS, INPUT_DIM)
    
    #------------------------------------------------------#
    #   获得模型并进行训练。
    #------------------------------------------------------#
    model = get_attention_model(TIME_STEPS, INPUT_DIM)
    model.summary()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X, Y, epochs=5, batch_size=64, validation_split=0.1)

    attention_vectors = []
    #------------------------------------------------------#
    #   取三百个样本，将他们通道的平均注意力情况取出来
    #------------------------------------------------------#
    for i in range(300):
        testing_X, testing_Y = get_data_recurrent(1, TIME_STEPS, INPUT_DIM)
        attention_vector = get_activations(model,testing_X,layer_name='attention_vec')
        print('attention =', attention_vector)
        assert (np.sum(attention_vector) - 1.0) < 1e-5
        attention_vectors.append(attention_vector)
    attention_vector_final = np.mean(np.array(attention_vectors), axis=0)

    #------------------------------------------------------#
    #   将结果绘制成图
    #------------------------------------------------------#
    pd.DataFrame(attention_vector_final, columns=['attention (%)']).plot(kind='bar',title='Attention Mechanism as a function of input dimensions.')
    plt.show()
