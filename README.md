# IS160_AI_In_Business_PetraMaciel
## Collection of labs completed in IS160, AI in Business course. 

## LAB 1: Manual Calculation of DL

Lab 1 was on manually calculating Deep Learning with our partners. The DL calculation example in the Michael Tayler book was used. We put down weight and bias weight calculations, summations, the activation function, total error, and gradients. Below are images of our written work on the calculation.

![My Image](lab1.1.png)
![My Image](lab1.2.png)
![My Image](lab1.3jpg.png)
![My Image](lab1.4.png)
![My Image](lab1.5.png)
![My Image](lab1.6.png)

## LAB 2: Deep Learning CNN with Dot Product

Lab 2 focused on viewing the notebook given to us and then changing parts to get our desired outcome. We then had to create a Dot Product with our values to check we did them correctly.
import numpy as np

Petra Maciel and Trinh Pham

# Forward Propagation using Dot Product

input_data = np.array([2,3])
weights = {'node_0':np.array([1,1]),
           'node_1':np.array([-1,1]),
           'output':np.array([2,-1])}
node_0_value = (input_data*weights['node_0']).sum()
node_1_value = (input_data*weights['node_1']).sum()


hidden_layer_values = np.array([node_0_value, node_1_value])
print(hidden_layer_values)

output = (hidden_layer_values * weights['output']).sum()
print(output)


Petra Maciel and Trinh Pham

```
# This is formatted as code
```

input_data = np.array([1,2,3])
weights = {'node_2':np.array([-1,1,2]),
           'node_3' :np.array([1,-1,3]),
           'node_4' :np.array([2,3,4]),
           'output':np.array([2,-2,3])}
node_2_value = (input_data*weights['node_2']).sum()
node_3_value = (input_data*weights['node_3']).sum()
node_4_value = (input_data*weights['node_4']).sum()

hidden_layer_values = np.array([node_2_value, node_3_value, node_4_value])
print(hidden_layer_values)

output = (hidden_layer_values * weights['output']).sum()
print(output)

## Try with a different model structure


Trinh Pham and Yadira Heng

We added more nodes and changed the numbers to make it more complex. We then drew it out on paper.

## Lab 3: MNIST Data

In Lab 3 we had to create two models, the MNIST model from the Chollet book, and an MNIST model from ChatGPT. This was to practice our ability to use GPT to our advantage. 
Petra Maciel and Jesus Maciel Barragan

Loading the MNIST dataset in Keras

from tensorflow.keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

Adding the training data to associate the images and labels

train_images.shape

len(test_labels)

test_labels

Setting the network architecture

from tensorflow import keras
from tensorflow.keras import layers
model = keras.Sequential([layers.Dense(512, activation="relu"), layers.Dense(10, activation="softmax")])

Reshaping the data into a shape the model expects

model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

Preparing the image data

train_images = train_images.reshape((60000, 28*28))
train_images = train_images.astype("float32")/255
test_images = test_images.reshape((10000, 28*28))
test_images = test_images.astype("float32")/255

fitting the model to the training data

model.fit(train_images, train_labels, epochs=5, batch_size=128)

we use the model to make predictions

test_digits = test_images[0:10]
predictions = model.predict(test_digits)
predictions[0]

predictions[0].argmax()

predictions[0][7]

test_labels[0]

evaluating the model on new data and check test accuracy

test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"test_acc:{test_acc}")

CHATGPT

# Import necessary libraries
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist

# Load the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Preprocess the data
# Reshape the data to fit the input shape of the neural network and normalize it to values between 0 and 1
train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255

# Convert labels to one-hot encoded format
train_labels = tf.keras.utils.to_categorical(train_labels)
test_labels = tf.keras.utils.to_categorical(test_labels)

# Define the neural network model
model = models.Sequential()
# Add a Flatten layer to convert the 2D image data into a 1D array
model.add(layers.Flatten(input_shape=(28, 28, 1)))
# Add a Dense layer with 128 units and ReLU activation function
model.add(layers.Dense(128, activation='relu'))
# Add a Dense layer with 10 units (for 10 classes) and softmax activation function for classification
model.add(layers.Dense(10, activation='softmax'))

# Compile the model
# Use the Adam optimizer, categorical crossentropy as the loss function, and track accuracy as a metric
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
# Use the training data and labels, batch size of 32, and train for 5 epochs
model.fit(train_images, train_labels, batch_size=32, epochs=5)

# Evaluate the model on the test data
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_acc}")


## Lab 4: DL Exploration

For Lab 4, as a pair, we created two models. The first is the IMDB data DL model from the Chollet book and then we collected three datasets of our choice from Kaggle or any sources we like for the second model. We tried different loss functions and optimizers for comparison and added comments within our codes.
Petra Maciel and Jesus Maciel

---
**🚨 Important Note 🚨**

If you've run one of the models in this notebook and wish to run a subsequent model, please **restart the runtime first**. Running multiple models consecutively without restarting can lead to memory issues and may cause the Colab session to crash.

To restart the runtime:
- Click on the `Runtime` menu at the top.
- Select `Restart runtime...`.
- After restarting, you can proceed with running the next model.

Thank you for your understanding!
---


---
**📘 Notebook Contents 📘**

This notebook presents the original model with the dataset used in the referenced book. In addition, we have expanded the notebook to include three other datasets sourced from Kaggle:

1. **Twitter Data**
2. **Stock Data**
3. **Climate Data**

Each dataset is accompanied by a model in a distinct cell. When you run a cell, you execute the entire model for that specific dataset.

To avoid memory or runtime issues, it's recommended to run only one model at a time and restart the runtime before proceeding with the next model, as noted in the previous disclaimer.


---



## **Original dataset in book**

# The IMDB dataset

print("Loading the IMDB dataset...")
# Loading the IMDB dataset
from tensorflow.keras.datasets import imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(
    num_words=10000)

# Decoding reviews back to text
print("\nDecoding the first review back to text...")
word_index = imdb.get_word_index()
reverse_word_index = dict(
    [(value, key) for (key, value) in word_index.items()])
decoded_review = " ".join(
    [reverse_word_index.get(i - 3, "?") for i in train_data[0]])
print(decoded_review)

print("\nPreparing the data...")
# Preparing the data
# Encoding the integer sequences via multi-hot encoding
import numpy as np
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        for j in sequence:
            results[i, j] = 1.
    return results
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)
y_train = np.asarray(train_labels).astype("float32")
y_test = np.asarray(test_labels).astype("float32")


print("\nBuilding the model...")
# Building your model
# Model definition
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Dense(16, activation="relu"),
    layers.Dense(16, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])


# Compiling the model
model.compile(optimizer="rmsprop",
              loss="binary_crossentropy",
              metrics=["accuracy"])

print("\nSetting aside a validation set...")
# Validating your approach
# Setting aside a validation set
x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]


print("\nTraining the model...")
# Training your model
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val, y_val))
history_dict = history.history
print("Keys in history dict:", history_dict.keys())

print("\nPlotting the training and validation loss...")
# Plotting the training and validation loss
import matplotlib.pyplot as plt
history_dict = history.history
loss_values = history_dict["loss"]
val_loss_values = history_dict["val_loss"]
epochs = range(1, len(loss_values) + 1)
plt.plot(epochs, loss_values, "bo", label="Training loss")
plt.plot(epochs, val_loss_values, "b", label="Validation loss")
plt.title("Training and validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

print("\nPlotting the training and validation accuracy...")
# Plotting the training and validation accuracy
plt.clf()
acc = history_dict["accuracy"]
val_acc = history_dict["val_accuracy"]
plt.plot(epochs, acc, "bo", label="Training acc")
plt.plot(epochs, val_acc, "b", label="Validation acc")
plt.title("Training and validation accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

print("\nRetraining a model from scratch...")
# Retraining a model from scratch
model = keras.Sequential([
    layers.Dense(16, activation="relu"),
    layers.Dense(16, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])
model.compile(optimizer="rmsprop",
              loss="binary_crossentropy",
              metrics=["accuracy"])
model.fit(x_train, y_train, epochs=4, batch_size=512)
results = model.evaluate(x_test, y_test)

print("\nEvaluation results:", results)

print("\nGenerating predictions on test data...")
# Using a trained model to generate predictions on new data
predictions = model.predict(x_test)
print(predictions)


# *Twitter Data*
- Using Adam optimizer
- Using hinge loss function

The dataset has three sentiments namely, negative(-1), neutral(0), and positive(+1). It contains two fields for the tweet and label.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("Importing dataset using pandas...")
# import dataset using pandas
df = pd.read_csv("/content/Twitter_Data.csv")

print("\nCleaning up the dataset...")
# remove neutral responses
df = df[df['category'] != 0.0]
# drop missing values
df = df.dropna(subset=['clean_text'])
df = df.dropna(subset=['category'])
# Reset index as the above
df.reset_index(drop=True, inplace=True)
# replace -1.0 with 0 for negative responses
df['category'] = df['category'].replace(-1.0, 0)
# replace 1.0 with 1 for positive responses
df['category'] = df['category'].replace(1.0, 1)

print("\nInitializing a tokenizer...")
# Initialize a tokenizer
from tensorflow.keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer(num_words=10000)  # Consider the top 10,000 words in the dataset

# Fit the tokenizer on the clean_text column
tokenizer.fit_on_texts(df['clean_text'])

print("\nEncoding the clean_text column...")
# Convert the clean_text column into integer sequences and put it in new column named endcoded_text
df['encoded_text'] = tokenizer.texts_to_sequences(df['clean_text'])

print("\nSelecting relevant columns...")
# select only the category and encoded text columns
df = df[['category', 'encoded_text']]

print("\nSplitting the data into training and testing sets...")
# Use sklearn library to split data
from sklearn.model_selection import train_test_split

# Splitting the data into training and testing sets
train_data, test_data, train_labels, test_labels = train_test_split(
    df['encoded_text'].tolist(),
    df['category'].tolist(),
    test_size=0.2,
    random_state=42
)

print("\nConverting the data to numpy arrays...")
# Convert the data to numpy arrays
train_data = np.array(train_data, dtype=object)
test_data = np.array(test_data)
train_labels = np.array(train_labels, dtype=object)
test_labels = np.array(test_labels)

print("\nFiltering out empty sequences from training and testing sets...")
# Filtering out empty sequences
filtered_data, filtered_labels = zip(*[(seq, label) for seq, label in zip(train_data, train_labels) if seq])
# Convert back to lists
train_data = list(filtered_data)
train_labels = list(filtered_labels)

# Filtering out empty sequences
filtered_data, filtered_labels = zip(*[(seq, label) for seq, label in zip(test_data, test_labels) if seq])
# Convert back to lists
test_data = list(filtered_data)
test_labels = list(filtered_labels)

print("\nEncoding the integer sequences via multi-hot encoding...")
# **Encoding the integer sequences via multi-hot encoding**
import numpy as np
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        for j in sequence:
            results[i, j] = 1.
    return results
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

y_train = np.asarray(train_labels).astype("float32")
y_test = np.asarray(test_labels).astype("float32")

# Check for NaN values in partial_x_train and partial_y_train
nan_in_x_train = np.isnan(x_train).any()
nan_in_y_train = np.isnan(y_train).any()

print("NaN in partial_x_train:", nan_in_x_train)
print("NaN in partial_y_train:", nan_in_y_train)

print("\nBuilding the model...")
# Building your model
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Dense(16, activation="relu"),
    layers.Dense(16, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])

print("\nCompiling the model...")
# compiling the model
model.compile(optimizer="adam",
              loss="hinge",
              metrics=["accuracy"])

print("\nSplitting the dataset into training and validation sets...")
# validating approach

# Reduce both the data and labels to match the original data's size
x_train = x_train[:25000]
y_train = y_train[:25000]

# Split the reduced dataset into training and validation sets
x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

print("\nTraining the model...")
# training your model:
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val, y_val))

print("\nPlotting training and validation loss...")
# plotting training and validation loss
history_dict = history.history
loss_values = history_dict["loss"]
val_loss_values = history_dict["val_loss"]
epochs = range(1, len(loss_values) + 1)
plt.plot(epochs, loss_values, "bo", label="Training loss")
plt.plot(epochs, val_loss_values, "b", label="Validation loss")
plt.title("Training and validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

print("\nPlotting the training and validation accuracy...")
# Plotting the training and validation accuracy
plt.clf()
acc = history_dict["accuracy"]
val_acc = history_dict["val_accuracy"]
plt.plot(epochs, acc, "bo", label="Training acc")
plt.plot(epochs, val_acc, "b", label="Validation acc")
plt.title("Training and validation accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

print("\nRetraining a model from scratch...")
# Retraining a model from scratch
model = keras.Sequential([
    layers.Dense(16, activation="relu"),
    layers.Dense(16, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])
model.compile(optimizer="adam",
              loss="hinge",
              metrics=["accuracy"])
model.fit(x_train, y_train, epochs=4, batch_size=512)
results = model.evaluate(x_test, y_test)

# look at results of trained model
print("\nEvaluation results:", results)

print("\nGenerating predictions on test data...")
# Using a trained model to generate predictions on new data
predictions = model.predict(x_test)
print(predictions)


# *Stock Data*
- Using SGD (Stochastic Gradient Descent) optimizer
- Using squared_hinge loss function

Gathered Stock news from Multiple twitter Handles regarding Economic news dividing into two parts : Negative(-1) and positive(1) .
* Negative count: 2,106
* Positive count: 3,685

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("Importing dataset using pandas...")
# import dataset using pandas
df = pd.read_csv("/content/stock_data.csv")
print("Unique Sentiments in the dataset:", df['Sentiment'].unique())

print("\nCleaning up the dataset...")
# drop missing values
df = df.dropna(subset=['Sentiment'])
df = df.dropna(subset=['Text'])
# Reset index as the above
df.reset_index(drop=True, inplace=True)
# replace -1.0 with 0 for negative responses
df['Sentiment'] = df['Sentiment'].replace(-1.0, 0)

print("\nInitializing a tokenizer...")
# Initialize a tokenizer
from tensorflow.keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer(num_words=10000)  # Consider the top 10,000 words in the dataset

print("\nFitting the tokenizer on the Text column...")
# Fit the tokenizer on the clean_text column
tokenizer.fit_on_texts(df['Text'])

print("\nEncoding the Text column...")
# Convert the clean_text column into integer sequences and put it in new column named endcoded_text
df['encoded_text'] = tokenizer.texts_to_sequences(df['Text'])

print("\nSelecting relevant columns...")
# select only the category and encoded text columns
df = df[['Sentiment', 'encoded_text']]

print("\nSplitting the data into training and testing sets...")
# Use sklearn library to split data
from sklearn.model_selection import train_test_split

# Splitting the data into training and testing sets
train_data, test_data, train_labels, test_labels = train_test_split(
    df['encoded_text'].tolist(),
    df['Sentiment'].tolist(),
    test_size=0.2,
    random_state=42
)

print("\nConverting the data to numpy arrays...")
# Convert the data to numpy arrays
train_data = np.array(train_data, dtype=object)
test_data = np.array(test_data)
train_labels = np.array(train_labels, dtype=object)
test_labels = np.array(test_labels)

print("\nFiltering out empty sequences from training and testing sets...")
# Filtering out empty sequences
filtered_data, filtered_labels = zip(*[(seq, label) for seq, label in zip(train_data, train_labels) if seq])
# Convert back to lists
train_data = list(filtered_data)
train_labels = list(filtered_labels)

# Filtering out empty sequences
filtered_data, filtered_labels = zip(*[(seq, label) for seq, label in zip(test_data, test_labels) if seq])
# Convert back to lists
test_data = list(filtered_data)
test_labels = list(filtered_labels)

print("\nEncoding the integer sequences via multi-hot encoding...")
# **Encoding the integer sequences via multi-hot encoding**
import numpy as np
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        for j in sequence:
            results[i, j] = 1.
    return results
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

y_train = np.asarray(train_labels).astype("float32")
y_test = np.asarray(test_labels).astype("float32")

# Check for NaN values in partial_x_train and partial_y_train
nan_in_x_train = np.isnan(x_train).any()
nan_in_y_train = np.isnan(y_train).any()

print("NaN in partial_x_train:", nan_in_x_train)
print("NaN in partial_y_train:", nan_in_y_train)

print("\nBuilding the model...")
# Building your model
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Dense(16, activation="relu"),
    layers.Dense(16, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])

print("\nCompiling the model...")
# compiling the model
model.compile(optimizer="sgd",
              loss="squared_hinge",
              metrics=["accuracy"])

print("\nSplitting the dataset into training and validation sets...")
# This part was changed from the twitter model
# Calculate the number of samples in the dataset
total_samples = len(x_train)

# Calculate the split index for an 80-20 split
split_index = int(0.8 * total_samples)

# Split the dataset into training and validation sets
partial_x_train = x_train[:split_index]
x_val = x_train[split_index:]
partial_y_train = y_train[:split_index]
y_val = y_train[split_index:]

print("\nTraining the model...")
# training your model:
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val, y_val))

print("\nPlotting training and validation loss...")
# plotting training and validation loss
history_dict = history.history
loss_values = history_dict["loss"]
val_loss_values = history_dict["val_loss"]
epochs = range(1, len(loss_values) + 1)
plt.plot(epochs, loss_values, "bo", label="Training loss")
plt.plot(epochs, val_loss_values, "b", label="Validation loss")
plt.title("Training and validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

print("\nPlotting the training and validation accuracy...")
# Plotting the training and validation accuracy
plt.clf()
acc = history_dict["accuracy"]
val_acc = history_dict["val_accuracy"]
plt.plot(epochs, acc, "bo", label="Training acc")
plt.plot(epochs, val_acc, "b", label="Validation acc")
plt.title("Training and validation accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

print("\nRetraining a model from scratch...")
# Retraining a model from scratch
model = keras.Sequential([
    layers.Dense(16, activation="relu"),
    layers.Dense(16, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])
model.compile(optimizer="rmsprop",
              loss="binary_crossentropy",
              metrics=["accuracy"])
model.fit(x_train, y_train, epochs=4, batch_size=512)
results = model.evaluate(x_test, y_test)

# look at results of trained model
print("\nEvaluation results:", results)

print("\nGenerating predictions on test data...")
# Using a trained model to generate predictions on new data
predictions = model.predict(x_test)
print(predictions)


# *Climate Data*
- Using ftrl optimizer
- Using Mean Absolute Error (MAE) Loss function

This dataset aggregates tweets pertaining to climate change collected between Apr 27, 2015 and Feb 21, 2018. In total, 43943 tweets were annotated. Each tweet is labelled independently by 3 reviewers. This dataset only contains tweets that all 3 reviewers agreed on (the rest were discarded).

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("Importing climate dataset using pandas...")
# import dataset using pandas
df = pd.read_csv("/content/climate_data.csv")

print("\nCleaning up the dataset...")
# drop tweet id column
df = df.drop("tweetid", axis=1)

# remove rows with 2(News): the tweet links to factual news about climate change
df = df[df['sentiment'] != 2]
# remove rows with 0(Neutral: the tweet neither supports nor refutes the belief of man-made climate change
df = df[df['sentiment'] != 0.0]
# drop missing values
df = df.dropna(subset=['sentiment'])
df = df.dropna(subset=['message'])
# Reset index as the above
df.reset_index(drop=True, inplace=True)
# replace -1.0 with 0 for negative responses(does not believe in climate change)
df['sentiment'] = df['sentiment'].replace(-1.0, 0)

print("\nInitializing a tokenizer...")
# Initialize a tokenizer
from tensorflow.keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer(num_words=10000)  # Consider the top 10,000 words in the dataset

print("\nFitting the tokenizer on the message column...")
# Fit the tokenizer on the message column
tokenizer.fit_on_texts(df['message'])

print("\nEncoding the message column...")
# Convert the message column into integer sequences and put it in new column named endcoded_text
df['encoded_text'] = tokenizer.texts_to_sequences(df['message'])

print("\nSelecting relevant columns...")
# select only the category and encoded text columns
df = df[['sentiment', 'encoded_text']]

print("\nSplitting the data into training and testing sets...")
# Use sklearn library to split data
from sklearn.model_selection import train_test_split

# Splitting the data into training and testing sets
train_data, test_data, train_labels, test_labels = train_test_split(
    df['encoded_text'].tolist(),
    df['sentiment'].tolist(),
    test_size=0.2,
    random_state=42
)

print("\nConverting the data to numpy arrays...")
# Convert the data to numpy arrays
train_data = np.array(train_data, dtype=object)
test_data = np.array(test_data)
train_labels = np.array(train_labels, dtype=object)
test_labels = np.array(test_labels)

print("\nFiltering out empty sequences from training and testing sets...")
# Filtering out empty sequences
filtered_data, filtered_labels = zip(*[(seq, label) for seq, label in zip(train_data, train_labels) if seq])
# Convert back to lists
train_data = list(filtered_data)
train_labels = list(filtered_labels)

# Filtering out empty sequences
filtered_data, filtered_labels = zip(*[(seq, label) for seq, label in zip(test_data, test_labels) if seq])
# Convert back to lists
test_data = list(filtered_data)
test_labels = list(filtered_labels)

print("\nEncoding the integer sequences via multi-hot encoding...")
# **Encoding the integer sequences via multi-hot encoding**
import numpy as np
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        for j in sequence:
            results[i, j] = 1.
    return results
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

y_train = np.asarray(train_labels).astype("float32")
y_test = np.asarray(test_labels).astype("float32")

# Check for NaN values in partial_x_train and partial_y_train
nan_in_x_train = np.isnan(x_train).any()
nan_in_y_train = np.isnan(y_train).any()

print("NaN in partial_x_train:", nan_in_x_train)
print("NaN in partial_y_train:", nan_in_y_train)

print("\nBuilding the model...")
# Building your model
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Dense(16, activation="relu"),
    layers.Dense(16, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])

print("\nCompiling the model...")
# compiling the model
model.compile(optimizer="ftrl",
              loss="mean_absolute_error",
              metrics=["accuracy"])

print("\nSplitting the dataset into training and validation sets...")
# Reduce both the data and labels to match the original data's size
x_train = x_train[:25000]
y_train = y_train[:25000]

# Split the reduced dataset into training and validation sets
x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

print("\nTraining the model...")
# training your model:
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val, y_val))

print("\nPlotting training and validation loss...")
# plotting training and validation loss
history_dict = history.history
loss_values = history_dict["loss"]
val_loss_values = history_dict["val_loss"]
epochs = range(1, len(loss_values) + 1)
plt.plot(epochs, loss_values, "bo", label="Training loss")
plt.plot(epochs, val_loss_values, "b", label="Validation loss")
plt.title("Training and validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

print("\nPlotting the training and validation accuracy...")
# Plotting the training and validation accuracy
plt.clf()
acc = history_dict["accuracy"]
val_acc = history_dict["val_accuracy"]
plt.plot(epochs, acc, "bo", label="Training acc")
plt.plot(epochs, val_acc, "b", label="Validation acc")
plt.title("Training and validation accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

print("\nRetraining a model from scratch...")
# Retraining a model from scratch
model = keras.Sequential([
    layers.Dense(16, activation="relu"),
    layers.Dense(16, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])
model.compile(optimizer="ftrl",
              loss="mean_absolute_error",
              metrics=["accuracy"])
model.fit(x_train, y_train, epochs=4, batch_size=512)
results = model.evaluate(x_test, y_test)

# look at results of trained model
print("\nEvaluation results:", results)

print("\nGenerating predictions on test data...")
# Using a trained model to generate predictions on new data
predictions = model.predict(x_test)
print(predictions)


## Lab 5: Big City Taxi Fare DL Model

In Lab 5, we displayed some formulas and data on the whiteboard. We had to pretend to write down how we would code on paper. We had to use three different dates and times as well.

## Lab 6: Big City Taxi Fare DL Model

For Lab 6, we had to actually put our written code from Lab 5 into our computers. We used the same formulas and data provided to us.
Petra Maciel and Jesus Maciel

---
**🚨 Important Note 🚨**

If you've run one of the models in this notebook and wish to run a subsequent model, please **restart the runtime first**. Running multiple models consecutively without restarting can lead to memory issues and may cause the Colab session to crash.

To restart the runtime:
- Click on the `Runtime` menu at the top.
- Select `Restart runtime...`.
- After restarting, you can proceed with running the next model.

Thank you for your understanding!
---


---
**📘 Notebook Contents 📘**

This notebook presents the original model with the dataset used in the referenced book. In addition, we have expanded the notebook to include three other datasets sourced from Kaggle:

1. **Twitter Data**
2. **Stock Data**
3. **Climate Data**

Each dataset is accompanied by a model in a distinct cell. When you run a cell, you execute the entire model for that specific dataset.

To avoid memory or runtime issues, it's recommended to run only one model at a time and restart the runtime before proceeding with the next model, as noted in the previous disclaimer.


---



## **Original dataset in book**

# The IMDB dataset

print("Loading the IMDB dataset...")
# Loading the IMDB dataset
from tensorflow.keras.datasets import imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(
    num_words=10000)

# Decoding reviews back to text
print("\nDecoding the first review back to text...")
word_index = imdb.get_word_index()
reverse_word_index = dict(
    [(value, key) for (key, value) in word_index.items()])
decoded_review = " ".join(
    [reverse_word_index.get(i - 3, "?") for i in train_data[0]])
print(decoded_review)

print("\nPreparing the data...")
# Preparing the data
# Encoding the integer sequences via multi-hot encoding
import numpy as np
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        for j in sequence:
            results[i, j] = 1.
    return results
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)
y_train = np.asarray(train_labels).astype("float32")
y_test = np.asarray(test_labels).astype("float32")


print("\nBuilding the model...")
# Building your model
# Model definition
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Dense(16, activation="relu"),
    layers.Dense(16, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])


# Compiling the model
model.compile(optimizer="rmsprop",
              loss="binary_crossentropy",
              metrics=["accuracy"])

print("\nSetting aside a validation set...")
# Validating your approach
# Setting aside a validation set
x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]


print("\nTraining the model...")
# Training your model
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val, y_val))
history_dict = history.history
print("Keys in history dict:", history_dict.keys())

print("\nPlotting the training and validation loss...")
# Plotting the training and validation loss
import matplotlib.pyplot as plt
history_dict = history.history
loss_values = history_dict["loss"]
val_loss_values = history_dict["val_loss"]
epochs = range(1, len(loss_values) + 1)
plt.plot(epochs, loss_values, "bo", label="Training loss")
plt.plot(epochs, val_loss_values, "b", label="Validation loss")
plt.title("Training and validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

print("\nPlotting the training and validation accuracy...")
# Plotting the training and validation accuracy
plt.clf()
acc = history_dict["accuracy"]
val_acc = history_dict["val_accuracy"]
plt.plot(epochs, acc, "bo", label="Training acc")
plt.plot(epochs, val_acc, "b", label="Validation acc")
plt.title("Training and validation accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

print("\nRetraining a model from scratch...")
# Retraining a model from scratch
model = keras.Sequential([
    layers.Dense(16, activation="relu"),
    layers.Dense(16, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])
model.compile(optimizer="rmsprop",
              loss="binary_crossentropy",
              metrics=["accuracy"])
model.fit(x_train, y_train, epochs=4, batch_size=512)
results = model.evaluate(x_test, y_test)

print("\nEvaluation results:", results)

print("\nGenerating predictions on test data...")
# Using a trained model to generate predictions on new data
predictions = model.predict(x_test)
print(predictions)


# *Twitter Data*
- Using Adam optimizer
- Using hinge loss function

The dataset has three sentiments namely, negative(-1), neutral(0), and positive(+1). It contains two fields for the tweet and label.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("Importing dataset using pandas...")
# import dataset using pandas
df = pd.read_csv("/content/Twitter_Data.csv")

print("\nCleaning up the dataset...")
# remove neutral responses
df = df[df['category'] != 0.0]
# drop missing values
df = df.dropna(subset=['clean_text'])
df = df.dropna(subset=['category'])
# Reset index as the above
df.reset_index(drop=True, inplace=True)
# replace -1.0 with 0 for negative responses
df['category'] = df['category'].replace(-1.0, 0)
# replace 1.0 with 1 for positive responses
df['category'] = df['category'].replace(1.0, 1)

print("\nInitializing a tokenizer...")
# Initialize a tokenizer
from tensorflow.keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer(num_words=10000)  # Consider the top 10,000 words in the dataset

# Fit the tokenizer on the clean_text column
tokenizer.fit_on_texts(df['clean_text'])

print("\nEncoding the clean_text column...")
# Convert the clean_text column into integer sequences and put it in new column named endcoded_text
df['encoded_text'] = tokenizer.texts_to_sequences(df['clean_text'])

print("\nSelecting relevant columns...")
# select only the category and encoded text columns
df = df[['category', 'encoded_text']]

print("\nSplitting the data into training and testing sets...")
# Use sklearn library to split data
from sklearn.model_selection import train_test_split

# Splitting the data into training and testing sets
train_data, test_data, train_labels, test_labels = train_test_split(
    df['encoded_text'].tolist(),
    df['category'].tolist(),
    test_size=0.2,
    random_state=42
)

print("\nConverting the data to numpy arrays...")
# Convert the data to numpy arrays
train_data = np.array(train_data, dtype=object)
test_data = np.array(test_data)
train_labels = np.array(train_labels, dtype=object)
test_labels = np.array(test_labels)

print("\nFiltering out empty sequences from training and testing sets...")
# Filtering out empty sequences
filtered_data, filtered_labels = zip(*[(seq, label) for seq, label in zip(train_data, train_labels) if seq])
# Convert back to lists
train_data = list(filtered_data)
train_labels = list(filtered_labels)

# Filtering out empty sequences
filtered_data, filtered_labels = zip(*[(seq, label) for seq, label in zip(test_data, test_labels) if seq])
# Convert back to lists
test_data = list(filtered_data)
test_labels = list(filtered_labels)

print("\nEncoding the integer sequences via multi-hot encoding...")
# **Encoding the integer sequences via multi-hot encoding**
import numpy as np
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        for j in sequence:
            results[i, j] = 1.
    return results
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

y_train = np.asarray(train_labels).astype("float32")
y_test = np.asarray(test_labels).astype("float32")

# Check for NaN values in partial_x_train and partial_y_train
nan_in_x_train = np.isnan(x_train).any()
nan_in_y_train = np.isnan(y_train).any()

print("NaN in partial_x_train:", nan_in_x_train)
print("NaN in partial_y_train:", nan_in_y_train)

print("\nBuilding the model...")
# Building your model
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Dense(16, activation="relu"),
    layers.Dense(16, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])

print("\nCompiling the model...")
# compiling the model
model.compile(optimizer="adam",
              loss="hinge",
              metrics=["accuracy"])

print("\nSplitting the dataset into training and validation sets...")
# validating approach

# Reduce both the data and labels to match the original data's size
x_train = x_train[:25000]
y_train = y_train[:25000]

# Split the reduced dataset into training and validation sets
x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

print("\nTraining the model...")
# training your model:
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val, y_val))

print("\nPlotting training and validation loss...")
# plotting training and validation loss
history_dict = history.history
loss_values = history_dict["loss"]
val_loss_values = history_dict["val_loss"]
epochs = range(1, len(loss_values) + 1)
plt.plot(epochs, loss_values, "bo", label="Training loss")
plt.plot(epochs, val_loss_values, "b", label="Validation loss")
plt.title("Training and validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

print("\nPlotting the training and validation accuracy...")
# Plotting the training and validation accuracy
plt.clf()
acc = history_dict["accuracy"]
val_acc = history_dict["val_accuracy"]
plt.plot(epochs, acc, "bo", label="Training acc")
plt.plot(epochs, val_acc, "b", label="Validation acc")
plt.title("Training and validation accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

print("\nRetraining a model from scratch...")
# Retraining a model from scratch
model = keras.Sequential([
    layers.Dense(16, activation="relu"),
    layers.Dense(16, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])
model.compile(optimizer="adam",
              loss="hinge",
              metrics=["accuracy"])
model.fit(x_train, y_train, epochs=4, batch_size=512)
results = model.evaluate(x_test, y_test)

# look at results of trained model
print("\nEvaluation results:", results)

print("\nGenerating predictions on test data...")
# Using a trained model to generate predictions on new data
predictions = model.predict(x_test)
print(predictions)


# *Stock Data*
- Using SGD (Stochastic Gradient Descent) optimizer
- Using squared_hinge loss function

Gathered Stock news from Multiple twitter Handles regarding Economic news dividing into two parts : Negative(-1) and positive(1) .
* Negative count: 2,106
* Positive count: 3,685

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("Importing dataset using pandas...")
# import dataset using pandas
df = pd.read_csv("/content/stock_data.csv")
print("Unique Sentiments in the dataset:", df['Sentiment'].unique())

print("\nCleaning up the dataset...")
# drop missing values
df = df.dropna(subset=['Sentiment'])
df = df.dropna(subset=['Text'])
# Reset index as the above
df.reset_index(drop=True, inplace=True)
# replace -1.0 with 0 for negative responses
df['Sentiment'] = df['Sentiment'].replace(-1.0, 0)

print("\nInitializing a tokenizer...")
# Initialize a tokenizer
from tensorflow.keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer(num_words=10000)  # Consider the top 10,000 words in the dataset

print("\nFitting the tokenizer on the Text column...")
# Fit the tokenizer on the clean_text column
tokenizer.fit_on_texts(df['Text'])

print("\nEncoding the Text column...")
# Convert the clean_text column into integer sequences and put it in new column named endcoded_text
df['encoded_text'] = tokenizer.texts_to_sequences(df['Text'])

print("\nSelecting relevant columns...")
# select only the category and encoded text columns
df = df[['Sentiment', 'encoded_text']]

print("\nSplitting the data into training and testing sets...")
# Use sklearn library to split data
from sklearn.model_selection import train_test_split

# Splitting the data into training and testing sets
train_data, test_data, train_labels, test_labels = train_test_split(
    df['encoded_text'].tolist(),
    df['Sentiment'].tolist(),
    test_size=0.2,
    random_state=42
)

print("\nConverting the data to numpy arrays...")
# Convert the data to numpy arrays
train_data = np.array(train_data, dtype=object)
test_data = np.array(test_data)
train_labels = np.array(train_labels, dtype=object)
test_labels = np.array(test_labels)

print("\nFiltering out empty sequences from training and testing sets...")
# Filtering out empty sequences
filtered_data, filtered_labels = zip(*[(seq, label) for seq, label in zip(train_data, train_labels) if seq])
# Convert back to lists
train_data = list(filtered_data)
train_labels = list(filtered_labels)

# Filtering out empty sequences
filtered_data, filtered_labels = zip(*[(seq, label) for seq, label in zip(test_data, test_labels) if seq])
# Convert back to lists
test_data = list(filtered_data)
test_labels = list(filtered_labels)

print("\nEncoding the integer sequences via multi-hot encoding...")
# **Encoding the integer sequences via multi-hot encoding**
import numpy as np
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        for j in sequence:
            results[i, j] = 1.
    return results
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

y_train = np.asarray(train_labels).astype("float32")
y_test = np.asarray(test_labels).astype("float32")

# Check for NaN values in partial_x_train and partial_y_train
nan_in_x_train = np.isnan(x_train).any()
nan_in_y_train = np.isnan(y_train).any()

print("NaN in partial_x_train:", nan_in_x_train)
print("NaN in partial_y_train:", nan_in_y_train)

print("\nBuilding the model...")
# Building your model
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Dense(16, activation="relu"),
    layers.Dense(16, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])

print("\nCompiling the model...")
# compiling the model
model.compile(optimizer="sgd",
              loss="squared_hinge",
              metrics=["accuracy"])

print("\nSplitting the dataset into training and validation sets...")
# This part was changed from the twitter model
# Calculate the number of samples in the dataset
total_samples = len(x_train)

# Calculate the split index for an 80-20 split
split_index = int(0.8 * total_samples)

# Split the dataset into training and validation sets
partial_x_train = x_train[:split_index]
x_val = x_train[split_index:]
partial_y_train = y_train[:split_index]
y_val = y_train[split_index:]

print("\nTraining the model...")
# training your model:
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val, y_val))

print("\nPlotting training and validation loss...")
# plotting training and validation loss
history_dict = history.history
loss_values = history_dict["loss"]
val_loss_values = history_dict["val_loss"]
epochs = range(1, len(loss_values) + 1)
plt.plot(epochs, loss_values, "bo", label="Training loss")
plt.plot(epochs, val_loss_values, "b", label="Validation loss")
plt.title("Training and validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

print("\nPlotting the training and validation accuracy...")
# Plotting the training and validation accuracy
plt.clf()
acc = history_dict["accuracy"]
val_acc = history_dict["val_accuracy"]
plt.plot(epochs, acc, "bo", label="Training acc")
plt.plot(epochs, val_acc, "b", label="Validation acc")
plt.title("Training and validation accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

print("\nRetraining a model from scratch...")
# Retraining a model from scratch
model = keras.Sequential([
    layers.Dense(16, activation="relu"),
    layers.Dense(16, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])
model.compile(optimizer="rmsprop",
              loss="binary_crossentropy",
              metrics=["accuracy"])
model.fit(x_train, y_train, epochs=4, batch_size=512)
results = model.evaluate(x_test, y_test)

# look at results of trained model
print("\nEvaluation results:", results)

print("\nGenerating predictions on test data...")
# Using a trained model to generate predictions on new data
predictions = model.predict(x_test)
print(predictions)


# *Climate Data*
- Using ftrl optimizer
- Using Mean Absolute Error (MAE) Loss function

This dataset aggregates tweets pertaining to climate change collected between Apr 27, 2015 and Feb 21, 2018. In total, 43943 tweets were annotated. Each tweet is labelled independently by 3 reviewers. This dataset only contains tweets that all 3 reviewers agreed on (the rest were discarded).

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("Importing climate dataset using pandas...")
# import dataset using pandas
df = pd.read_csv("/content/climate_data.csv")

print("\nCleaning up the dataset...")
# drop tweet id column
df = df.drop("tweetid", axis=1)

# remove rows with 2(News): the tweet links to factual news about climate change
df = df[df['sentiment'] != 2]
# remove rows with 0(Neutral: the tweet neither supports nor refutes the belief of man-made climate change
df = df[df['sentiment'] != 0.0]
# drop missing values
df = df.dropna(subset=['sentiment'])
df = df.dropna(subset=['message'])
# Reset index as the above
df.reset_index(drop=True, inplace=True)
# replace -1.0 with 0 for negative responses(does not believe in climate change)
df['sentiment'] = df['sentiment'].replace(-1.0, 0)

print("\nInitializing a tokenizer...")
# Initialize a tokenizer
from tensorflow.keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer(num_words=10000)  # Consider the top 10,000 words in the dataset

print("\nFitting the tokenizer on the message column...")
# Fit the tokenizer on the message column
tokenizer.fit_on_texts(df['message'])

print("\nEncoding the message column...")
# Convert the message column into integer sequences and put it in new column named endcoded_text
df['encoded_text'] = tokenizer.texts_to_sequences(df['message'])

print("\nSelecting relevant columns...")
# select only the category and encoded text columns
df = df[['sentiment', 'encoded_text']]

print("\nSplitting the data into training and testing sets...")
# Use sklearn library to split data
from sklearn.model_selection import train_test_split

# Splitting the data into training and testing sets
train_data, test_data, train_labels, test_labels = train_test_split(
    df['encoded_text'].tolist(),
    df['sentiment'].tolist(),
    test_size=0.2,
    random_state=42
)

print("\nConverting the data to numpy arrays...")
# Convert the data to numpy arrays
train_data = np.array(train_data, dtype=object)
test_data = np.array(test_data)
train_labels = np.array(train_labels, dtype=object)
test_labels = np.array(test_labels)

print("\nFiltering out empty sequences from training and testing sets...")
# Filtering out empty sequences
filtered_data, filtered_labels = zip(*[(seq, label) for seq, label in zip(train_data, train_labels) if seq])
# Convert back to lists
train_data = list(filtered_data)
train_labels = list(filtered_labels)

# Filtering out empty sequences
filtered_data, filtered_labels = zip(*[(seq, label) for seq, label in zip(test_data, test_labels) if seq])
# Convert back to lists
test_data = list(filtered_data)
test_labels = list(filtered_labels)

print("\nEncoding the integer sequences via multi-hot encoding...")
# **Encoding the integer sequences via multi-hot encoding**
import numpy as np
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        for j in sequence:
            results[i, j] = 1.
    return results
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

y_train = np.asarray(train_labels).astype("float32")
y_test = np.asarray(test_labels).astype("float32")

# Check for NaN values in partial_x_train and partial_y_train
nan_in_x_train = np.isnan(x_train).any()
nan_in_y_train = np.isnan(y_train).any()

print("NaN in partial_x_train:", nan_in_x_train)
print("NaN in partial_y_train:", nan_in_y_train)

print("\nBuilding the model...")
# Building your model
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Dense(16, activation="relu"),
    layers.Dense(16, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])

print("\nCompiling the model...")
# compiling the model
model.compile(optimizer="ftrl",
              loss="mean_absolute_error",
              metrics=["accuracy"])

print("\nSplitting the dataset into training and validation sets...")
# Reduce both the data and labels to match the original data's size
x_train = x_train[:25000]
y_train = y_train[:25000]

# Split the reduced dataset into training and validation sets
x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

print("\nTraining the model...")
# training your model:
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val, y_val))

print("\nPlotting training and validation loss...")
# plotting training and validation loss
history_dict = history.history
loss_values = history_dict["loss"]
val_loss_values = history_dict["val_loss"]
epochs = range(1, len(loss_values) + 1)
plt.plot(epochs, loss_values, "bo", label="Training loss")
plt.plot(epochs, val_loss_values, "b", label="Validation loss")
plt.title("Training and validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

print("\nPlotting the training and validation accuracy...")
# Plotting the training and validation accuracy
plt.clf()
acc = history_dict["accuracy"]
val_acc = history_dict["val_accuracy"]
plt.plot(epochs, acc, "bo", label="Training acc")
plt.plot(epochs, val_acc, "b", label="Validation acc")
plt.title("Training and validation accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

print("\nRetraining a model from scratch...")
# Retraining a model from scratch
model = keras.Sequential([
    layers.Dense(16, activation="relu"),
    layers.Dense(16, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])
model.compile(optimizer="ftrl",
              loss="mean_absolute_error",
              metrics=["accuracy"])
model.fit(x_train, y_train, epochs=4, batch_size=512)
results = model.evaluate(x_test, y_test)

# look at results of trained model
print("\nEvaluation results:", results)

print("\nGenerating predictions on test data...")
# Using a trained model to generate predictions on new data
predictions = model.predict(x_test)
print(predictions)


## Lab 7: Prompt Eng. - Stable Diffusion

In Lab 7, we used Paperspace to start a notebook and upload a provided kernel for stable diffusion. We ran all cells in order to get the link we needed to access it. As a pair, we decided on images we wanted to create using AI and gave descriptive prompts to get our desired results. 

## Lab 8: Image to Image - Stable Diffusion

Similar to the previous lab, Lab 8 focused on editing images we already had and inputting them to create more customized images we wanted. We used and edited four different images and compared them to each of their pairs. We used features such as inpaint, denoiser function, and masking. 

## Lab 10 RL Diagram
As partners, we provided six different environments by defining and listing items such as the agent, action, environment, state, and reward.

## Lab 13: Responsible AI
Let’s say your group is an AI consulting group. How would you consult the companies that face
the following situations in their operations? If applicable, what specific coding statements
would you add or edit? Capture and summarize your group discussion.

## Lab 14: AI Implementation
We will explore the AI implementation lesson
and apply it to your group’s new AI-driven information system. You can build a system that
will utilize any of the algorithms (DL, RL, any of the ML algorithms).
Define your information system (eg. what type of system? where in the organization? who will
use it mostly?, what output/result can you expect?, who will fund this? How about ethics?
etc...)
For each of the development phase, list the major activities
Data (what kind data? where will you get the data? how will you address the feature
engineering, data train/test, etc)
Fine-tuning and implementation
Maintenance and fine-tuning (may need more data? new data??)
Addressing unforeseen conflicts/disasters
Implementing customer feedback to improve customer satisfaction
Turn in all your group work, capture the discussion as much as possible, do any flowchart
or visual illustrations and summaries.

## Summary

Throughout these labs, I have gained experience with AI and insight into how of many of these processes work and what they mean. It can be of benefit when needing to work with something involving AI because I have gotten a base on the concept and functionality. 
