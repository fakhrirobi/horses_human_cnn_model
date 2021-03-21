## Human and Horses Classifier : 
### Project Details : 
* This project Aim to implement Convolutional Neural Network to Classify Human and Horses Picture 
* The Classifier was implemented in Desktop GUI Using TkInter 


#### Python packages (version stated in requirements.txt):
* Tensorflow 
* Numpy 
* TkInter
* Os


### Project Step by Step : 


1. Training model ( training.py ) , model details : 
    #### 1). Neural Network Architecture :         
        Conv2D(16,(3,3),activation='relu',input_shape=(300,300,3)),
        MaxPooling2D(2,2),
        Conv2D(32,(3,3),activation='relu'),
        MaxPooling2D(2,2),
        Conv2D(64,(3,3),activation='relu'),
        MaxPooling2D(2,2),
        Conv2D(64,(3,3),activation='relu'),
        MaxPooling2D(2,2),
        Conv2D(64,(3,3),activation='relu'),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(512,activation='relu'),
        Dense(1,activation='sigmoid')
    #### 2). Image Augmentation : 
        ```
        ImageDataGenerator(
        rescale=1/255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

     #### 3). Number of Epochs = 10 


2. Saving Model (horse_human.h5)
   
3. Loading Model Adding GUI Using Tkinter (app.py) 


## How to use this project : 

1. Clone this project by entering this line of code in Command Prompt : 
   ```
   git clone https://github.com/fakhrirobi/horses_human_cnn_model.git
   ```
2. Move to directory of cloned repo by entering : 
   ```
   cd horses_human_cnn_model
   ```
3. Install the dependencies in requirements.txt by entering : 
   ```
   pip install -r requirements.txt
   ```
4. Run the app.py : 
   ```
   python app.py
   ```
5. The Tkinter Dialog Box Will Show Up 
   ![Tkiner Dialog Box](assets\tkinter_display.png)
6. Click Pick an Image! and the file explorer dialog will appear,and select the image you want to classify
   ![Tkiner Dialog Box](assets\tkinter_display.png)
7. 





