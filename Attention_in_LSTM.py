from keras.layers import merge
from keras.layers.core import *
from keras.layers.recurrent import LSTM
from keras.models import *
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


INPUT_DIM = 2
TIME_STEPS = 10

#-------------------------------------------#
#   对每一个step的INPUT_DIM的attention几率
#   求平均
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
#-------------------------------------------#
#   获得数据集
#   attention_column代表我们希望被注意的列
#-------------------------------------------#
def get_data_recurrent(n, time_steps, input_dim, attention_column=2):
    x = np.random.normal(loc=0, scale=10, size=(n, time_steps, input_dim))
    y = np.random.randint(low=0, high=2, size=(n, 1))
    x[:, attention_column, :] = np.tile(y[:], (1, input_dim))
    return x, y
#-------------------------------------------#
#   注意力模块
#-------------------------------------------#
def attention_3d_block(inputs):
    # inputs.shape = (batch_size, time_steps, lstm_units)

    # (batch_size, time_steps, lstm_units) -> (batch_size, lstm_units, time_steps)
    a = Permute((2, 1))(inputs)

    # 对最后一维进行全连接
    # (batch_size, lstm_units, time_steps) -> (batch_size, lstm_units, time_steps)
    a = Dense(TIME_STEPS, activation='softmax')(a)

    # (batch_size, lstm_units, time_steps) -> (batch_size, time_steps, lstm_units)
    a_probs = Permute((2, 1), name='attention_vec')(a)

    # 相乘
    # 相当于获得每一个step中，每个维度在所有step中的权重
    output_attention_mul = merge([inputs, a_probs], name='attention_mul', mode='mul')
    return output_attention_mul

#-------------------------------------------#
#  建立注意力模型
#-------------------------------------------#
def get_attention_model():
    inputs = Input(shape=(TIME_STEPS, INPUT_DIM,))
    lstm_units = 32
    # (batch_size, time_steps, INPUT_DIM) -> (batch_size, input_dim, lstm_units)
    lstm_out = LSTM(lstm_units, return_sequences=True)(inputs)
    attention_mul = attention_3d_block(lstm_out)
    # (batch_size, input_dim, lstm_units) -> (batch_size, input_dim*lstm_units)
    attention_mul = Flatten()(attention_mul)
    output = Dense(1, activation='sigmoid')(attention_mul)
    model = Model(input=[inputs], output=output)
    return model

if __name__ == '__main__':

    N = 100000

    X, Y = get_data_recurrent(N, TIME_STEPS, INPUT_DIM)

    model = get_attention_model()

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    print(model.summary())

    model.fit(X, Y, epochs=1, batch_size=64, validation_split=0.1)

    attention_vectors = []
    for i in range(300):
        testing_X, testing_Y = get_data_recurrent(1, TIME_STEPS, INPUT_DIM)
        attention_vector = get_activations(model,testing_X,layer_name='attention_vec')
        print('attention =', attention_vector)
        assert (np.sum(attention_vector) - 1.0) < 1e-5
        attention_vectors.append(attention_vector)

    attention_vector_final = np.mean(np.array(attention_vectors), axis=0)

    pd.DataFrame(attention_vector_final, columns=['attention (%)']).plot(kind='bar',
                                                                         title='Attention Mechanism as '
                                                                               'a function of input'
                                                                               ' dimensions.')
    plt.show()