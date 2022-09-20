# Article-Classification
classifying 70GB of scientific articles in both Persian and English languages in .docx and .pdf files.

The documentations of main packages are available in following links:
- NLTK: https://www.nltk.org/
- Pandas: https://pandas.pydata.org/docs/index.html
- xlwings: https://docs.xlwings.org/en/stable/
- TensorFlow: https://www.tensorflow.org/api_docs
- sklearn: https://scikit-learn.org/stable/


#Getting Started
One of the best RNNs is LSTM that can be used in a variety of cases, in this case we are using it for article classification which is a part of NLP tasks.
Our first challenge was that the data was in word and pdf format and we needed to extract data from them and put it next to the labels that were assigned to them. For the sake of memory we decided to only extract abstracts from each article and since abstracts are in the beginning of each article, extracting the first 500 words would be beneficial for this case. Then we put extracted clean data into another column and now our main project will be started.

```python
def txtCleaning(inputTXT):
    final = " "
    translator = str.maketrans(string.punctuation, ' '*len(string.punctuation)) #map punctuation to space
    punctuationfree=inputTXT.translate(translator)
    
    # print(punctuationfree)
    punctuationfreelower = [i.lower() for i in punctuationfree]
    # print(punctuationfreelower)
    final = ''.join(map(str,punctuationfreelower))
    # print(final)
    # tokens = word_tokenize(final)
    # # print(tokens)
    final = " ".join([i for i in final.split() if i not in stopwords])
    # # print(output)
    # lemm_text = [wordnet_lemmatizer.lemmatize(word) for word in output]
    # # print(lemm_text)
    # finalStr = ' '.join(lemm_text)

    # return lemm_text
    return final
```

# LSTM
Many parameter initializations are done using the interpretation of validation_loss/training_loss. A good model with no overfitting or underfitting happens when both losses are decreasing and  validation_loss is above training_loss.
in our case because of the lack of the clean balanced data and an enormous number of classes (about 48) our best result happened in an overfitting case with 43% accuracy.

the initialize parameters are given as below:
```python
model = tf.keras.Sequential([
    tf.keras.Input(shape=(1,), dtype=tf.string),
    encoder,
    Embedding(
        input_dim=len(encoder.get_vocabulary()),
        output_dim=120,
        mask_zero=True),
    
    Bidirectional(LSTM(200, return_sequences=False)),
    Dropout(0.5),
    BatchNormalization(),
    Dense(64, activation='relu',kernel_regularizer='l1'),
    Dense(48,activation='softmax')
    
])
```

```python
model.compile(loss="sparse_categorical_crossentropy",
              optimizer=tf.keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),
              metrics=['accuracy'])
```

# Requierments
Python 3.6+ is required. The following packages are required:

- [Pandas](https://pandas.pydata.org/docs/)
- [TensorFlow](https://www.tensorflow.org/api_docs)
- [sklearn](https://scikit-learn.org/stable/)

# Notes
Plot interpretations and LSTM algorithm references are listed bellow:
- https://developers.google.com/machine-learning/testing-debugging/metrics/interpretic
- https://colah.github.io/posts/2015-08-Understanding-LSTMs/




