import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.utils import np_utils

filename = "sonnet_text.txt"


text = (open(filename).read()).lower()


uniquechar = sorted(list(set(text)))

char2int = {}
int2char = {}

for i, c in enumerate(uniquechar):
    char2int.update({c: i})
    int2char.update({i: c})


X = []
Y = []
for i in range(0, len(text) - 40, 1):
    sequence = text[i:i + 40]
    label = text[i + 40]
    X.append([char2int[char] for char in sequence])
    Y.append(char2int[label])

    

##Reshaping of X
X_reshaped = numpy.reshape(X, (len(X), 40,1))
X_reshaped = X_reshaped / float(len(uniquechar))
Y_reshaped = np_utils.to_categorical(Y, num_classes = None)


# defining the LSTM model
model = Sequential()
model.add(LSTM(300, input_shape=(X_reshaped.shape[1], X_reshaped.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(300))
model.add(Dropout(0.2))
model.add(Dense(Y_reshaped.shape[1], activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam')



#Fitting
model.fit(X_reshaped, Y_reshaped, epochs=4, batch_size=30)

start_index = numpy.random.randint(0,len(X)-1) ####taking seed randomely
new_string = X[start_index]



##character generation
for i in range(40):
     x = numpy.reshape(new_string, (1, len(new_string), 1))
     x = x / float(len(uniquechar))

#######
     pred_index = numpy.argmax(model.predict(x, verbose=0))
     char_out = int2char[pred_index]
     seq_in = [int2char[value] for value in new_string]
     print(char_out)

     new_string.append(pred_index)
     new_string = new_string[1:len(new_string)]



######

     
     




