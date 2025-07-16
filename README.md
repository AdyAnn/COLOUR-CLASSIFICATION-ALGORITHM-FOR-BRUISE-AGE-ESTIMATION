from google.colab import drive
import os

drive.mount ('/content/drive/')


base_dir = '/content/drive/My Drive/FGA/'
!ls "/content/drive/My Drive/FGA/"

from google.colab import drive
drive.mount('/content/drive')

train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
evaluation_dir = os.path.join(base_dir, 'evaluation')

red_dir = os.path.join(train_dir, 'red/')
purple_dir = os.path.join(train_dir, 'purple/')
yellow_dir = os.path.join(train_dir, 'yellow/')

print("Count on each class of data train")
print('Count on red bruises : ', len(os.listdir(red_dir)))
print('Count on purple bruises : ', len(os.listdir(purple_dir)))
print('Count on yellow bruises : ', len(os.listdir(yellow_dir)))

train_red = os.path.join(train_dir, 'red/')
train_purple = os.path.join(train_dir, 'purple/')
train_yellow = os.path.join(train_dir, 'yellow/')

validation_red = os.path.join(validation_dir, 'red/')
validation_purple = os.path.join(validation_dir, 'purple/')
validation_yellow = os.path.join(validation_dir, 'yellow/')

evaluation_red = os.path.join(evaluation_dir, 'red/')
evaluation_purple = os.path.join(evaluation_dir, 'purple/')
evaluation_yellow = os.path.join(evaluation_dir, 'yellow/')

red_dir = 'my_dataset/red/'


!pip install albumentations --quiet


import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2
from tensorflow.keras.utils import Sequence, to_categorical
import numpy as np
import os


class AlbumentationDataGenerator(Sequence):
    def __init__(self, image_paths, labels, batch_size, augmentations, input_size=(150, 150), n_classes=3, shuffle=True):
        self.image_paths = image_paths
        self.labels = labels
        self.batch_size = batch_size
        self.augmentations = augmentations
        self.input_size = input_size
        self.shuffle = shuffle
        self.n_classes = n_classes
        self.indexes = np.arange(len(image_paths))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.image_paths) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        batch_paths = [self.image_paths[k] for k in indexes]
        batch_labels = [self.labels[k] for k in indexes]

        images = []
        for img_path in batch_paths:
            image = cv2.imread(img_path)
            image = cv2.cvtColour(image, cv2.COLOUR_BGR2RGB)
            image = cv2.resize(image, self.input_size)
            augmented = self.augmentations(image=image)
            image = augmented['image']
            images.append(image)

        X = np.array(images, dtype=np.float32) / 255.0
        y = to_categorical(batch_labels, num_classes=self.n_classes)
        return X, y


train_transform = A.Compose([
    A.RandomBrightnessContrast(p=0.5),
    A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=20, p=0.4),
    A.RandomGamma(p=0.3),
])

val_transform = A.Compose([
    A.RandomBrightnessContrast(p=0.2),
    A.HueSaturationValue(p=0.2),
    A.Resize(150, 150)
])


from glob import glob

def get_image_paths_and_labels(folder):
    image_paths = []
    labels = []
    class_names = sorted(os.listdir(folder))
    label_map = {name: idx for idx, name in enumerate(class_names)}

    for class_name in class_names:
        class_folder = os.path.join(folder, class_name)
        for fname in os.listdir(class_folder):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(class_folder, fname))
                labels.append(label_map[class_name])
    return image_paths, labels


train_paths, train_labels = get_image_paths_and_labels(train_dir)
val_paths, val_labels = get_image_paths_and_labels(validation_dir)
eval_paths, eval_labels = get_image_paths_and_labels(evaluation_dir)

train_generator = AlbumentationDataGenerator(train_paths, train_labels, batch_size=10, augmentations=train_transform)
val_generator = AlbumentationDataGenerator(val_paths, val_labels, batch_size=10, augmentations=val_transform, shuffle=False)
eval_generator = AlbumentationDataGenerator(eval_paths, eval_labels, batch_size=10, augmentations=val_transform, shuffle=False)


import os
import random
from shutil import copyfile

# Function to ensure a directory exists
def ensure_dir_exists(path):
    os.makedirs(path, exist_ok=True)

# Function to split dataset into train, validation, and eval
def train_val_eval_split(source, train, val, eval_dir, train_ratio, val_ratio):
    total_files = os.listdir(source)
    total_size = len(total_files)

    # Shuffle files
    randomized = random.sample(total_files, total_size)

    # Calculate split sizes
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    eval_size = total_size - train_size - val_size  # remainder

    # Split file list
    train_files = randomized[:train_size]
    val_files = randomized[train_size:train_size + val_size]
    eval_files = randomized[train_size + val_size:]

    # Copy files
    for file in train_files:
        copyfile(os.path.join(source, file), os.path.join(train, file))
    for file in val_files:
        copyfile(os.path.join(source, file), os.path.join(val, file))
    for file in eval_files:
        copyfile(os.path.join(source, file), os.path.join(eval_dir, file))


# === Define all your directories ===
# Replace these with your actual paths
red_dir = os.path.join(train_dir, 'red/')
train_red = 'data_split/train/red/'
validation_red = 'data_split/val/red/'
eval_red = 'data_split/eval/red/'

purple_dir = os.path.join(train_dir, 'purple/')
train_purple = 'data_split/train/purple/'
validation_purple = 'data_split/val/purple/'
eval_purple = 'data_split/eval/purple/'

yellow_dir = os.path.join(train_dir, 'yellow/')
train_yellow = 'data_split/train/yellow/'
validation_yellow = 'data_split/val/yellow/'
eval_yellow = 'data_split/eval/yellow/'

# === Create all needed directories ===
all_dirs = [
    train_red, validation_red, eval_red,
    train_purple, validation_purple, eval_purple,
    train_yellow, validation_yellow, eval_yellow
]
for directory in all_dirs:
    ensure_dir_exists(directory)

# === Set split ratios ===
train_ratio = 0.7
val_ratio = 0.2  # eval will be 0.1 (automatically)

# === Run the split for each colour category ===
train_val_eval_split(red_dir, train_red, validation_red, eval_red, train_ratio, val_ratio)
train_val_eval_split(purple_dir, train_purple, validation_purple, eval_purple, train_ratio, val_ratio)
train_val_eval_split(yellow_dir, train_yellow, validation_yellow, eval_yellow, train_ratio, val_ratio)


from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# Add to model.fit(..., callbacks=[early_stop])




import tensorflow as tf

lr_schedule = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-6,
    verbose=1
)


def get_image_paths_and_labels(folder):
    image_paths = []
    labels = []
    class_names = sorted(os.listdir(folder))
    label_map = {name: idx for idx, name in enumerate(class_names)}

    for class_name in class_names:
        class_folder = os.path.join(folder, class_name)
        for img in os.listdir(class_folder):
            image_paths.append(os.path.join(class_folder, img))
            labels.append(label_map[class_name])

    return image_paths, labels, class_names  

train_paths, train_labels, class_names = get_image_paths_and_labels(train_dir)
val_paths, val_labels, _ = get_image_paths_and_labels(validation_dir)
eval_paths, eval_labels, _ = get_image_paths_and_labels(evaluation_dir)


print('Total of overall data of red bruises :', len(os.listdir(red_dir)))
print('Total of train red bruises :', len(os.listdir(train_red)))
print('Total of val red bruises :', len(os.listdir(validation_red)))
print('Total of eval red bruises :', len(os.listdir(evaluation_red)))

print('Total of overall data of purple bruises :', len(os.listdir(purple_dir)))
print('Total of train purple bruises :', len(os.listdir(train_purple)))
print('Total of val purple bruises :', len(os.listdir(validation_purple)))
print('Total of eval purple bruises :', len(os.listdir(eval_purple)))

print('Total of overall data of yellow bruises :', len(os.listdir(yellow_dir)))
print('Total of tra yellow bruises :', len(os.listdir(train_yellow)))
print('Total of val yellow bruises :', len(os.listdir(validation_yellow)))
print('Total of eval yellow bruises :', len(os.listdir(evaluation_yellow)))



import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import numpy as np # Moved numpy import here
import matplotlib.pyplot as plt # Also moving matplotlib here since it's used in tr_plot

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs = {}):
    if (logs.get('accuracy') > 0.85):
      print('\nAccuracy targeted up to 85%')
      self.model.stop_training = True

callbacks = myCallback()

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Load the base model (without top layers)
base_model = MobileNetV2(input_shape=(150, 150, 3), include_top=False, weights='imagenet')
base_model.trainable = False  # Freeze for faster training; unfreeze later for fine-tuning

# Add custom classification layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.4)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
predictions = Dense(3, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Compile
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()


model.summary()

tf.keras.layers.GlobalAveragePooling2D()


# Unfreeze the base model for fine-tuning
base_model.trainable = True

# Recompile with a lower learning rate
model.compile(optimizer=Adam(learning_rate=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])


validation_steps = len(val_generator)  # or use math.ceil(val_samples / batch_size)


history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),  # or a fixed number
    epochs=5,
    validation_data=val_generator,
    validation_steps=len(val_generator),
    verbose=1,
    callbacks=[callbacks]
)


import numpy as np
import matplotlib.pyplot as plt

def tr_plot(tr_data, start_epoch):
    # Plot the training and validation data
    tacc = tr_data.history['accuracy']
    tloss = tr_data.history['loss']
    vacc = tr_data.history['val_accuracy']
    vloss = tr_data.history['val_loss']

    Epoch_count = len(tacc) + start_epoch
    Epochs = []
    for i in range(start_epoch, Epoch_count):
        Epochs.append(i + 1)

    index_loss = np.argmin(vloss)  # epoch with the lowest val loss
    val_lowest = vloss[index_loss]

    index_acc = np.argmax(vacc)  # epoch with the highest val accuracy
    acc_highest = vacc[index_acc]

    plt.style.use('fivethirtyeight')
    sc_label = 'best epoch= ' + str(index_loss + 1 + start_epoch)
    vc_label = 'best epoch= ' + str(index_acc + 1 + start_epoch)

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 8))

    # Loss plot
    axes[0].plot(Epochs, tloss, 'r', label='Training loss')
    axes[0].plot(Epochs, vloss, 'g', label='Validation loss')
    axes[0].scatter(index_loss + 1 + start_epoch, val_lowest, s=150, c='blue', label=sc_label)
    axes[0].set_title('Training and Validation Loss')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].legend()

    # Accuracy plot
    axes[1].plot(Epochs, tacc, 'r', label='Training Accuracy')
    axes[1].plot(Epochs, vacc, 'g', label='Validation Accuracy')
    axes[1].scatter(index_acc + 1 + start_epoch, acc_highest, s=150, c='blue', label=vc_label)
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()

    plt.tight_layout()
    plt.show()


import math

validation_batch_size = 32
validation_steps = math.ceil(len(val_generator))



evaluation_batch_size = 32

# If using ImageDataGenerator or any generator that has .samples
if hasattr(eval_generator, 'samples'):
    evaluation_steps = math.ceil(eval_generator.samples / evaluation_batch_size)
else:
    evaluation_steps = len(eval_generator)


tr_plot(history, 0)

# Evaluate model
acc = model.evaluate(eval_generator, batch_size=evaluation_batch_size, verbose=1, steps=evaluation_steps)[1] * 100
train_loss = model.evaluate(eval_generator, steps=evaluation_steps, verbose=1)[0]


from IPython.display import Markdown, display

def print_md(msg):
    display(Markdown(msg))

print_md(f"✅ **Accuracy on the test set:** `{acc:.2f}%`")
print_md(f"📉 **Final Training Loss:** `{train_loss:.4f}`")




from keras.callbacks import ModelCheckpoint

import pandas as pd # Keep pandas here as it's not used in tr_plot
from sklearn.model_selection import train_test_split

from matplotlib.pyplot import imshow # Keep imshow here as it's not used in tr_plot
import os
import seaborn as sns
sns.set_style('darkgrid')
from sklearn.metrics import confusion_matrix, classification_report

def print_info( test_datagen, preds, print_code, save_dir, subject ):
    class_dict=test_datagen.class_indices
    labels= test_datagen.labels
    file_names= test_datagen.filenames
    error_list=[]
    true_class=[]
    pred_class=[]
    prob_list=[]
    new_dict={}
    error_indices=[]
    y_pred=[]
    for key,value in class_dict.items():
        new_dict[value]=key             # dictionary {integer of class number: string of class name}
    # store new_dict as a text fine in the save_dir
    classes=list(new_dict.values())     # list of string of class names
    dict_as_text=str(new_dict)
    dict_name= subject + '-' +str(len(classes)) +'.txt'
    dict_path=os.path.join(save_dir,dict_name)
    with open(dict_path, 'w') as x_file:
        x_file.write(dict_as_text)
    errors=0
    for i, p in enumerate(preds):
        pred_index=np.argmax(p)
        true_index=labels[i]  # labels are integer values
        if pred_index != true_index: # a misclassification has occurred
            error_list.append(file_names[i])
            true_class.append(new_dict[true_index])
            pred_class.append(new_dict[pred_index])
            prob_list.append(p[pred_index])
            error_indices.append(true_index)
            errors=errors + 1
        y_pred.append(pred_index)
    if print_code !=0:
        if errors>0:
            if print_code>errors:
                r=errors
            else:
                r=print_code
            msg='{0:^28s}{1:^28s}{2:^28s}{3:^16s}'.format('Filename', 'Predicted Class' , 'True Class', 'Probability')
            print(msg, (0,255,0),(55,65,80))
            for i in range(r):
                split1=os.path.split(error_list[i])
                split2=os.path.split(split1[0])
                fname=split2[1] + '/' + split1[1]
                msg='{0:^28s}{1:^28s}{2:^28s}{3:4s}{4:^6.4f}'.format(fname, pred_class[i],true_class[i], ' ', prob_list[i])
                print(msg, (255,255,255), (55,65,60))
                #print(error_list[i]  , pred_class[i], true_class[i], prob_list[i])
        else:
            msg='With accuracy of 100 % there are no errors to print'
            print(msg, (0,255,0),(55,65,80))
    if errors>0:
        plot_bar=[]
        plot_class=[]
        for  key, value in new_dict.items():
            count=error_indices.count(key)
            if count!=0:
                plot_bar.append(count) # list containg how many times a class c had an error
                plot_class.append(value)   # stores the class
        fig=plt.figure()
        fig.set_figheight(len(plot_class)/3)
        fig.set_figwidth(10)
        plt.style.use('fivethirtyeight')
        for i in range(0, len(plot_class)):
            c=plot_class[i]
            x=plot_bar[i]
            plt.barh(c, x, )
            plt.title( ' Errors by Class on Test Set')
    y_true= np.array(labels)
    y_pred=np.array(y_pred)
    if len(classes)<= 30:
        # create a confusion matrix
        cm = confusion_matrix(y_true, y_pred )
        length=len(classes)
        if length<8:
            fig_width=8
            fig_height=8
        else:
            fig_width= int(length * .5)
            fig_height= int(length * .5)
        plt.figure(figsize=(fig_width, fig_height))
        sns.heatmap(cm, annot=True, vmin=0, fmt='g', cmap='Blues', cbar=False)
        plt.xticks(np.arange(length)+.5, classes, rotation= 90)
        plt.yticks(np.arange(length)+.5, classes, rotation=0)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.show()
    clr = classification_report(y_true, y_pred, target_names=classes)
    print("Classification Report:\n----------------------\n", clr)

model.summary()

import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns


def print_info(generator, predictions, print_code=0, save_dir=None, subject=None):
    # True labels from the generator
    y_true = generator.classes

    # Class indices to label mapping (optional)
    class_indices = generator.class_indices
    index_to_label = dict((v, k) for k, v in class_indices.items())

    # Predicted class labels
    y_pred = np.argmax(predictions, axis=1)  # use for softmax output

    # Optionally print/save some basic info
    if print_code > 0:
        print(f"Class indices: {class_indices}")
        print(f"Sample actual: {y_true[:10]}")
        print(f"Sample predicted: {y_pred[:10]}")

    return y_true, y_pred



preds = model.predict(val_generator, verbose=1)


save_dir = "/content/drive/My Drive/FGA/results/"
subject = "bruise_classification"

def print_info(generator, predictions, print_code=0, save_dir=None, subject=None):
    # Use .labels from custom AlbumentationDataGenerator
    y_true = np.array(generator.labels)

    # Predicted labels (from softmax)
    y_pred = np.argmax(predictions, axis=1)

    # Optionally preview some outputs
    if print_code > 0:
        print("Sample Actual:", y_true[:10])
        print("Sample Predicted:", y_pred[:10])

    return y_true, y_pred

   y_true, y_pred = print_info(val_generator, preds, print_code=5, save_dir=save_dir, subject=subject)


class_names = ["Class Red", "Class Purple", "Class Yellow"]


cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted Labels")
plt.ylabel("Actual Labels")
plt.title("Confusion Matrix")
plt.show()


print("\nClassification Report (rows = actual classes):")
print(classification_report(y_true, y_pred, target_names=class_names))


import os

model_name = 'Bruise'  # Defined correctly as a string
save_dir = r'./'  # Save directory

# Ensure `subject` and `acc` are defined before using them
subject = 'test_subject'  # Example subject name
acc = 0.987  # Example accuracy value

# Corrected save_id with model_name
save_id = str(model_name + '-' + subject + '-' + str(acc)[:str(acc).rfind('.')+3] + '.h5')

save_loc = os.path.join(save_dir, save_id)

# Assuming `model` is defined (e.g., a trained Keras/TensorFlow model)
# model.save(save_loc)

print("Model saved at:", save_loc)

import numpy as np
import os
import matplotlib.pyplot as plt
from keras.preprocessing import image
from google.colab import files

# Upload image
uploaded = files.upload()

# Define class names (replace with your actual class labels)
class_list = ["Red bruise (1-2 days)", "yellow bruise (6-10 days)", "purple bruise (3-5 days)"]  # Modify as per your training classes

# Prepare lists to hold images and predictions
images = []
titles = []

for fn in uploaded.keys():
    path = fn
    img = image.load_img(path, target_size=(150, 150))

    # Preprocess image
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x /= 255.0

    # Predict class
    classes = model.predict(x)
    predicted_class_index = np.argmax(classes, axis=1)[0]
    predicted_class = class_list[predicted_class_index]

    # Save for later display
    images.append(img)
    titles.append(f"Predicted: {predicted_class}")

num_images = len(images)
rows = num_images
cols = 1

plt.figure(figsize=(6, rows * 4))  # Adjust height per image

for i in range(num_images):
    plt.subplot(rows, cols, i + 1)
    plt.imshow(images[i])
    plt.title(titles[i])
    plt.axis("off")

plt.tight_layout()
plt.show()
