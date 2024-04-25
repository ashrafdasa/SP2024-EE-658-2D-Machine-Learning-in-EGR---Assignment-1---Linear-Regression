import streamlit
import streamlit as st
import pickle
import os
from streamlit_drawable_canvas import st_canvas
import pandas as pd
from PIL import Image 
import numpy as np 
from sklearn import model_selection
import  streamlit_vertical_slider  as svs
import json
import warnings
warnings.filterwarnings("ignore")

def main():
    
    st.session_state.theme = "dark"
    
    st.title('Assignment-5')
    st.header('Ashraf Dasa') 
     
    col10, col20 = st.columns(2) # creating columns for UI
    with col10:   # the first column content 
        dataSetName = st.radio(  # define the options of the datasets 
            "Select Dataset: ",
            ('IRIS', 'Digits')) 
    with col20:  # the second column 
        alg = st.selectbox("Algorithm: ",  # defining the algorithms used 
                           [ 'None','Logistic Regression', 'Neural Network Classifier', "Naive Bayes","Decision Tree"])

    if(alg =="None"): # in case of no algorithem seleccted
        return;  # clear the page and end the application
        
    
    # at this point, the user selected the data set and selected the algortihm 
    scaler = pickle.load(open("models/scaling_"+dataSetName+".pkl", "rb"))  # get the scaling parameters for the selected dataset 
    loaded_model =  pickle.load( open("models/"+dataSetName+"_"+alg+".h5", 'rb') ) # get the model needed for the selected dataset and selected algorithm
    
    if(dataSetName == "IRIS"): # if user opt IRIS 
        col_IRIS_col10, col_IRIS_col20,col_IRIS_col30,col_IRIS_col40,col_IRIS_col50 = st.columns(5)  # Defin 5 columns for user interface
        with col_IRIS_col10:   # the first column
            SepalLength = svs.vertical_slider(key="Sepal Length",  # define a slider 
                    default_value=2, step=0.01, min_value=0, max_value=10) # ensure no humar error, by controling user input 
            st.write('Sepal Length')  # show the user the selected lable 
            st.write(SepalLength)  # show the user the selected value 
        with col_IRIS_col20: # the second column
            SepalWidth = svs.vertical_slider(key="Sepal Width",  # define a slider 
                    default_value=2,  step=0.01, min_value=0,  max_value=5,  )  # ensure no humar error, by controling user input 
            st.write('Sepal Width') # show the user the selected lable 
            st.write(SepalWidth)  # show the user the selected value 
        with col_IRIS_col30:
            PetalLength = svs.vertical_slider(key="Petal Length",  # define a slider 
                    default_value=2,  step=0.01, min_value=0,  max_value=10,  )  # ensure no humar error, by controling user input 
            st.write('Petal Length') # show the user the selected lable 
            st.write(PetalLength) # show the user the selected lable 
        with col_IRIS_col40:
            PetalWidth = svs.vertical_slider(key="Petal Width",  # define a slider 
                    default_value=2,  step=0.01,  min_value=0,  max_value=5,  )  # ensure no humar error, by controling user input 
            st.write('Petal Width') # show the user the selected lable 
            st.write(PetalWidth)  # show the user the selected lable 
        SubmitIRIS = st.button('Submit', key =  0) # submit the user selected values 
        
        if(SubmitIRIS): # if user clicks on submit 
            features = scaler.transform([[SepalLength,SepalWidth,PetalLength,PetalWidth]])   # build the input to the model  
            result = loaded_model.predict(features) # predict the values  
              
            if(result == 0): # read result 
                resultName = "setosa" # result name 
                resultImage = "setosa.jpg" # result image 
            elif(result == 1): # read result
                resultName = "versicolor" # result name 
                resultImage = "versicolor.jpg"
            else:  
                resultName = "virginica" # result name 
                resultImage = "Iris_virginica_2.jpg" # result image 
            with col_IRIS_col50: # show the result image in the last column
                st.image(resultImage, caption= resultName ) 
            
            st.metric(label="type of irise: ", value= resultName )  # show the iris name
            st.write( "Training Info : "  )  # show info 
            st.json(json.loads (open ("./models/"+dataSetName+"_"+alg+".json", "r").read()),expanded=False) # show the user training result if more info needed
        return

    else:
        col1, col2 = st.columns(2)    
        with col1:   # first col show the user input 
            tab1, tab2 = st.tabs(["Draw", "Upload" ])  # user can make input using 2 options 
            with tab1: 
                canvas_result = st_canvas( # defin canvas , user can draw a number 
                    fill_color="rgba(255, 165, 0, 0.3)",  stroke_width=30, 
                    stroke_color="#fff",   background_color= "#000",
                    background_image=None, height=160,
                    width=160,  drawing_mode="freedraw",  key="canvas", )  
                SubmitWriting = st.button('Submit',key=1)  # show submit botton to user once done drawing
            with tab2:  
                img_file_buffer = st.file_uploader('Upload a PNG image', type='png') # user can upload a file 
                SubmitUploadImage = st.button('Submit',key=2)  # show submit botton to user once file is selected 

        with col2: # second col show the user output   
            if(SubmitUploadImage or SubmitWriting):
                if(SubmitWriting and canvas_result.image_data is not None):   # if user submit draw option
                    img_array = canvas_result.image_data 
                    #im = Image.fromarray(img_array).convert('1') # convert image to grayscale 
                if(SubmitUploadImage and img_file_buffer is not None):     # if user submit upload file option
                    image = Image.open(img_file_buffer) # open loaded image
                    img_array = np.array(image)  # convert image to array  

                im = Image.fromarray(img_array).convert('1') # convert image to grayscale 
                im1 = im.resize((8, 8))  # resize image 
                flatImage = np.array(im1).flatten() * 255 # conver resized image to be 0 to 255 and flaten it to be in a single row
                flatImage = scaler.transform([flatImage])  # transform the image using the scaling factors 
                result = loaded_model.predict(flatImage)  # precit the image 
                
                st.metric(label="Expected Number ", value= result )     # print the result to the user 
                st.image(im1, caption='Rescaled Image')  # show part of the process, how the image is scaled down 
                st.write( "Training Info : " + alg)
                st.json(json.loads (open ("./models/"+dataSetName+"_"+alg+".json", "r").read()),expanded=False) # show the user training result if more info needed
                
        
main() # call the main function to start the application 