import numpy
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.layers import RepeatVector
from keras.callbacks import ModelCheckpoint

X=[]
f=open('KHKT2017\\save.txt','r',encoding='utf-8')
maxleny=int(f.readline())#số lượng món ăn trong 1 bữa
num2string=eval(f.readline())# ánh xạ từ số sang chuỗi
f.close()

f=open('KHKT2017\\test.txt','r',encoding='utf-8')
for line in f:
	st=line.split(';')
	X.append(st[:7])
f.close()
print(X)
def onehotvector(yy,numbest):
	xx=[]
	for i in range(numbest):
		xx.append(0)
	for x in range(len(yy)):
		xx[yy[x]]=1
	return xx

X = numpy.reshape(X, (len(X),len(X[0]),1))# chuyển qua 3D

model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256))
model.add(Dropout(0.2))
model.add(Dense(len(num2string), activation='softmax'))
filename = "KHKT2017\\weights-improvement-136-0.6066-bigger.hdf5"
model.load_weights(filename)
model.compile(loss='mean_squared_error', optimizer='RMSprop',metrics=['accuracy'])
prediction = model.predict(X,verbose=0)
print(prediction)
f=open('KHKT2017\\output.txt','w',encoding='utf-8')
for x in prediction:
	index = numpy.argmax(x)
	f.write(num2string[index]+'\n')
f.close()