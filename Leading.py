import os
import json
import pyttsx3
import numpy as np
import tkinter as Tk
from PIL import Image
import tensorflow as tf
from datetime import datetime
from tkinter import filedialog
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input

def load_dataset(directory):
    IMG_SIZE = (64, 64)
    data, labels = [], []
    class_mapping = {}    
    for label, category in enumerate(sorted(os.listdir(directory))):
        folder_path = os.path.join(directory, category)
        if not os.path.isdir(folder_path):
            continue
        class_mapping[label] = category
        for filename in os.listdir(folder_path):
            img_path = os.path.join(folder_path, filename)
            try:
                img = Image.open(img_path).convert("L")  # Grayscale
                img = img.resize(IMG_SIZE)
                img = np.array(img) / 255.0
                data.append(img)
                labels.append(label)
            except:
                continue  
    data = np.array(data).reshape(-1, 64, 64, 1)
    num_classes = len(class_mapping)
    input_shape = data.shape[1:]
    train_labels = to_categorical(labels, num_classes)
    return data, np.array(train_labels), class_mapping, input_shape, num_classes

def clear_screen(): 
    os.system('cls' if os.name == 'nt' else 'clear')
    
def preprocess_image(img_path, target_size=(64, 64)):
    img = Image.open(img_path).convert("L")
    img = img.resize(target_size)
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=-1)
    return img
def text_to_speech(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def get_greeting(use_utc=False):
    now = datetime.utcnow() if use_utc else datetime.now()
    hour = now.hour

    if 5 <= hour < 12:
        return "Good Morning!"
    elif 12 <= hour < 17:
        return "Good Afternoon!"
    else:
        return "Good Evening!"

while True:
    print("Test    The Model  For 1 :- ")
    print("Create  The Model  For 2 :- ")
    print("Delete  The Model  For 3 :- ")
    print("Clear   The Screen For * :- ")
    print("")
    print("Exit For 0 :- ")
    print("")
    n = input("Enter Your Choise :- ")
    if n=='1':
        root = Tk.Tk()
        root.withdraw()
        folder_path =  filedialog.askdirectory(title="Select a Model Folder")
        if not folder_path:
            continue
        files = os.listdir(folder_path)
        model_file = next((f for f in files if f.endswith(".keras")), None)
        mapping_file = next((f for f in files if f.endswith(".json")), None)
        if not model_file:
            print("Model File Does Not Exist")
        if not mapping_file:
            print("Mapping File Does Not Exist")            
        model_path = os.path.join(folder_path, model_file)
        mapping_path = os.path.join(folder_path, mapping_file)
        if os.path.exists(folder_path):
            print("Loading existing model...")
            model = tf.keras.models.load_model(model_path)
            num_classes = model.output_shape[-1]
            with open(mapping_path, "r") as f:
                mapping = json.load(f)
                mapping = {int(k): v for k, v in mapping.items()}
            wish = get_greeting(use_utc=False)
            print(wish)
            while True:
                root = Tk.Tk()
                root.withdraw()
                path =  filedialog.askopenfilename(title="Select a Image For Predict")
                if not path:
                    break
                else:
                    img = preprocess_image(path)
                    print(img.shape)
                    prediction = model.predict(img)
                    predicted_class = np.argmax(prediction)
                    print(predicted_class)
                    print("")
                    print("")
                    print("---------------------------------------------")
                    print(f"It's The {mapping[(predicted_class)]}.")
                    print("---------------------------------------------")
                    text_to_speech(f"It's The {mapping[(predicted_class)]}.")
                    print("")
                    print("")
        else:
            print("Model Not Exist")
    elif n=='2':
        root = Tk.Tk()
        root.withdraw()
        path =  filedialog.askdirectory(title="Select a Dataset")
        if path:
            train = int(input("How Many Times To Train The Model :-  "))
            print(f"{path} Data In Loading.........")
            data,labels,mapping,shape,classes = load_dataset(path)
            model = Sequential([Input(shape=(64, 64, 1)),
                                Conv2D(64, (3,3), activation='relu', padding='same'),
                                MaxPooling2D(2,2),
                                Conv2D(128, (3,3), activation='relu', padding='same'),
                                MaxPooling2D(2,2),
                                Conv2D(128, (3,3), activation='relu', padding='same'),
                                MaxPooling2D(2,2),
                                Conv2D(256, (3,3), activation='relu', padding='same'),
                                MaxPooling2D(2,2),
                                Conv2D(512, (3,3), activation='relu', padding='same'),
                                MaxPooling2D(2,2),
                                Conv2D(512, (3,3), activation='relu', padding='same'),
                                MaxPooling2D(2,2),    
                                Flatten(),
                                Dense(1024, activation='relu'),
                                Dense(classes, activation='softmax')])
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            model.fit(data,labels,epochs=train,batch_size=32)
            s = input(f"Model Are Ready With {len(data)} images and {classes} classes Do You Want To Save It Y/N !!.")
            if s=='y':
                name = input("Name For It :- ")
                folder_path = filedialog.askdirectory(title="Select Save Location")
                folder_path = os.path.join(folder_path,name + "_Model")
                os.makedirs(folder_path,exist_ok=True)
                path1 = os.path.join(folder_path,name + ".keras")
                mapp = os.path.join(folder_path,name + ".json")
                model.save(path1)
                with open(mapp, "w") as f:
                    json.dump(mapping, f)
                print(f"Model Are Stored In {folder_path} With {name} Name")
            elif s=='n':
                continue
    elif n=='3':
        folder_path = filedialog.askdirectory(title="Delet Model Directory")
        if folder_path:
            if os.path.exists(folder_path):
                shutil.rmtree(folder_path)
                print("Model Are deleted successfully!")
            else:
                print("Model Are are not found")
    elif n=='*':
        clear_screen()
    elif n=='0':
        break