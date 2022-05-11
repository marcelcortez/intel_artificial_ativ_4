import pandas as pd
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense

#Separa os dados
csv = pd.read_csv('dados.csv', sep=',')
csv = csv.drop(columns=['lote'])
le = LabelEncoder()
csv['fruta'] = le.fit_transform(csv['fruta'])
dados = csv.values
atributos = dados[:,1:]
classificadores = dados[:,0]

#Cria modelo e camadas da rede neural
modelo = Sequential()
modelo.add(Dense(units=5, activation='relu'))
modelo.add(Dense(units=1, activation='sigmoid'))

#Configura processamento
modelo.compile(optimizer='adam',  loss = 'binary_crossentropy', metrics = ['binary_accuracy'])
modelo.fit(atributos, classificadores, batch_size=10, epochs=500)

#Salva modelo
modelo.save('modelo.h5')