import requests
from bs4 import BeautifulSoup
import string
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences


def Web_scraping(root):

    website = f'{root}/movies'
    result = requests.get(website)
    content = result.text
    soup = BeautifulSoup(content, 'lxml')

    box = soup.find('article', class_='main-article')

    links = [link['href'] for link in box.find_all('a', href=True)]

    for link in links:
        result = requests.get(f'{root}/{link}')
        content = result.text
        soup = BeautifulSoup(content, 'lxml')

        box = soup.find('article', class_='main-article')
        title = box.find('h1').get_text()
        transcript = box.find(
            'div', class_='full-script').get_text(strip=True, separator=' ')
        i = 0
        while i < 10:
            with open('Script.txt', 'a', encoding="utf-8") as file:
                file.write(transcript)
            i = i + 1

#Web_scraping('https://subslikescript.com')


f = open('Script.txt','r',encoding = 'utf-8') 
line = f.readline()
#print(line)
#x = line.split(' ')
#print(len(x))


def cleanText(document):
    tokens = document.split()
    table = str.maketrans(' ',' ',string.punctuation)
    tokens = [w.translate(table) for w in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [word.lower() for word in tokens]
    return tokens

tokens = cleanText(line)
#print(len(set(tokens)))

length = 18 + 1

lines = []
for i in range(length,len(tokens)):
    seq = tokens[i - length : i]
    l = ' '.join(seq)
    lines.append(l)
    if i > 220000:
       break
#print(lines[0])
tokenizer = Tokenizer()
tokenizer.fit_on_texts(lines)
sequences = tokenizer.texts_to_sequences(lines)

sequences = np.array(sequences)
x,y = sequences[:, :-1],sequences[:,-1]

vocab_size = len(tokenizer.word_index) + 1
y = to_categorical(y,num_classes= vocab_size)

seq_length = x.shape[1]

model = Sequential()
model.add(Embedding(vocab_size,50, input_length=seq_length))
model.add(LSTM(100,return_sequences = True))
model.add(LSTM(100))
model.add(Dense(100,activation = 'relu'))
model.add(Dense(vocab_size,activation = 'softmax'))
model.summary()
model.compile(loss = 'categorical_crossentropy',optimizer= 'adam', metrics = ['accuracy'])
model.fit(x,y,batch_size= 256,epochs = 100 )
model.save('AI201_trainedModel')

#lines[123]

trained_model = keras.models.load_model("AI201_trainedModel")
def generateTextSequence(model,tokenizer, text_seq_length,seed_text, n_words):
  text = []

  for _ in range(n_words):
    encoded = tokenizer.texts_to_sequences([seed_text])[0]
    encoded = pad_sequences([encoded],maxlen = text_seq_length,truncating='pre')

    y_predict = model.predict_classes(encoded)
    predicted_word = ''
    for word,index in tokenizer.word_index.items():
      if index == y_predict:
        predicted_word = word
        predicted_word = word
        break
    seed_text = seed_text + ' '+ predicted_word
    text.append(predicted_word)
  return " ".join(text)

seed_text = input('Enter text for prediction: ')
print(generateTextSequence(trained_model,tokenizer,seq_length,seed_text, 10))

