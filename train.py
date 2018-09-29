import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.layers import RepeatVector
from keras.callbacks import ModelCheckpoint

X=[]
Y=[]
y=[]
string2num={}# ánh xạ từ chuỗi sang số
num2string={}# ánh xạ từ số sang chuỗi
maxleny=1#số lượng món ăn trong 1 bữa
count=0

f=open('KHKT2017\\data.csv','r',encoding='utf-8')
data=[]
for line in f:
	data.append(line)
f.close()
data=data[1:]#xóa dòng đầu tiên

def onehotvector(yy,numbest):
	xx=[]
	for i in range(numbest):
		xx.append(0)
	xx[yy]=1
	return xx

for i in range(len(data)):
	st=data[i].split(';')[:8]
	#print(i+1,' ',st)
	X.append(st[:7])#lấy 7 cột đầu tiên từ cột 0 đến cột 6
	Y.append(st[7].lower())# Chỉ lấy cột thứ 7 và chuyển về ký tự in thường
#print(X)
X = numpy.reshape(X, (len(X),len(X[0]),1))# chuyển qua 3D
#print(Y)
for st in Y:
	if st not in string2num:
		string2num[st]=count
		num2string[count]=st
		count+=1
	y.append(string2num[st])
#print(string2num)
#print(num2string)
#lưu lại các tham số để phục vụ cho việc test
f=open('KHKT2017\\save.txt','w',encoding='utf-8')
f.write(str(maxleny)+'\n')
f.write(str(num2string))
f.close()

for i in range(len(y)):
	y[i]=onehotvector(y[i],len(string2num))# chuyển thành dạng one-hot-vector
	#print(y[i])

model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256))
model.add(Dropout(0.2))
model.add(Dense(len(string2num), activation='softmax'))
model.compile(loss='mean_squared_error', optimizer='RMSprop',metrics=['accuracy'])
# define the checkpoint
filepath="KHKT2017\\weights-improvement-{epoch:02d}-{acc:.4f}-bigger.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
# fit the model
model.fit(X, y, epochs=150, batch_size=1, callbacks=callbacks_list)