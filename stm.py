
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf

st.title('face detection')
st.markdown('Toy model to play to classify iris flowers into \ setosa, versicolor, virginica')

file = st.file_uploader("Please upload an brain scan file", type=["jpg", "png"])

import cv2
from PIL import Image, ImageOps
import numpy as np
st.set_option('deprecation.showfileUploaderEncoding', False)
def import_and_predict(image_data):
    
        size = (180,180)    
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        image = np.asarray(image)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #img_resize = (cv2.resize(img, dsize=(75, 75),    interpolation=cv2.INTER_CUBIC))/255.
        
        img_reshape = img[np.newaxis,...]
        clf = joblib.load("rf_model.sav")
        prediction = clf.predict(img_reshape)
        
        return prediction
if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions = import_and_predict(image)
    score = tf.nn.softmax(predictions[0])
    st.write(prediction)
    st.write(score)
    st.text(predictions[0])
    print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
    

)