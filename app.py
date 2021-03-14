from tkinter import *
from tkinter import filedialog
import time
from time import sleep 
import os 
import tensorflow as tf 
import numpy as np 

#Handling CUBLAS fail to initialize
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

from tkinter import filedialog
LARGEFONT =("Montserrat", 20) 
NORMALFONT = ("Montserrat", 10)
class Path() :

    def file_path(self) : 
        global path
        file_path = filedialog.askopenfilename(title='Pick image you want to classify')
        path = file_path
        return path

class Model() : 
    def __init__(self) : 
        self.model = tf.keras.models.load_model('horse_human.h5')
    def classify(self) : 
        filename = [x for x in path.split('/')]
        filename = filename[-1]
        img = tf.keras.preprocessing.image.load_img(path,target_size=(300,300,3))
        x  = tf.keras.preprocessing.image.img_to_array(img)
        image_tensor = np.expand_dims(x,axis=0)
        classes = self.model.predict(image_tensor)
        prediction_text = list()
        if classes[0] > 0.5 : 
            prediction_text.append(filename+' is  detected as a human') 
        else : 
            prediction_text.append(filename+' is detected as a horse') 
        return prediction_text[0]
    # first window frame MainMenu 

class tkinterApp(Tk): 

    # __init__ function for class tkinterApp 
    def __init__(self, *args, **kwargs): 
        
        # __init__ function for class Tk 
        super().__init__(*args, **kwargs)
        
        # creating a container 
        container = Frame(self) 
        container.pack(side = "top", fill = "both", expand = True) 

        container.grid_rowconfigure(0, weight = 1) 
        container.grid_columnconfigure(0, weight = 1) 

        # initializing frames to an empty array 
        self.frames = {} 

        # iterating through a tuple consisting 
        # of the different page layouts 
        frame = MainMenu(container, self) 

            # initializing frame of that object from 
            # MainMenu, page1, page2 respectively with 
            # for loop 
        self.frames[MainMenu] = frame 

        frame.grid(row = 0, column = 0, sticky ="nsew") 

        self.show_frame(MainMenu) 

    # to display the current frame passed as 
    # parameter 
    def show_frame(self, cont): 
        frame = self.frames[cont] 
        frame.tkraise() 

class MainMenu(Frame): 
    def __init__(self, parent, controller): 
        Frame.__init__(self, parent) 
        PATH = Path()

        def show_result(): 
            model = Model()
            classification_result = model.classify()
            #check if the Output box already display prior result 
            prior_classification= Output.get("1.0", END)
            if prior_classification == '' :
                Output.insert(END, classification_result) 
            else :
                Output.delete("1.0", "end-1c") 
                Output.insert(END, classification_result) 


    # label of frame Layout 2 
        label = Label(self, text ="Horse and Human Classifier Using  Convolutional Neural Network ", font = LARGEFONT) 

        # putting the grid in its place by using 
        # grid 
        label.grid(row = 0, column = 1, padx = 10, pady = 10) 

        button1 = Button(self, text ='Pick an Image!',command = PATH.file_path,width=30,font=NORMALFONT) 
        button2 = Button(self,text='Show The Result', command = show_result,width=30,font=NORMALFONT)
        Output = Text(self, height = 5,  width = 30,font=NORMALFONT) 



        # putting the button in its place by 
        # using grid 
        button1.grid(row=1,column=1, sticky=W+E) 
        button2.grid(row=2,column=1, sticky=W+E)
        Output.grid(row=3,column=1, sticky=W+E)



if __name__ == '__main__':
    app = tkinterApp() 
    app.mainloop() 
