import streamlit as st
import pandas as pd
from PIL import Image
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
import pickle

# images and dictionaries to convert indices to image/label
image_setosa = Image.open("Classification_Iris_Setosa.jpg")
image_versicolor = Image.open("Classificatoin_Iris_Versicolor.jpg")
image_viginica = Image.open("Classification_Iris_Virginica.jpg")
prediction_image = {0:image_setosa, 1:image_versicolor, 2:image_viginica}
prediction_label = {0:"Setosa", 1:"Versicolor", 2:"Virginica"}

# configurate layout
st.set_page_config(layout="wide",initial_sidebar_state="expanded")

# title and description
st.title("Classification based on classic Iris flower set")
st.markdown("""
 This python application uses sklearn and a Random Forest Classifier to predict to which Iris category a flower with specified data belongs.
 The input on the sidebar will be used to predict a probability for each type.  
""")

expander_bar = st.beta_expander("About")
expander_bar.markdown("""
* **Python libraries: streamlit, pandas, PIL, sklearn, pickle**
* **Source of data: datasets.load_iris() in sklearn**
""")

# siderbar - sliders for input parameters of iris set
st.sidebar.header("User Input Parameters")

def user_input_features():
    sepal_length = st.sidebar.slider("Sepal length",4.0,8.0,7.0)
    sepal_width = st.sidebar.slider("Sepal width",2.0,5.0,3.5)
    petal_length = st.sidebar.slider("Petal length",1.0,7.0,2.0)
    petal_width = st.sidebar.slider("Petal width",0.1,2.5,0.5)
    data = {
        "Sepal length": sepal_length,
        "Sepal width":  sepal_width,
        "Petal length": petal_length,
        "Petal width":  petal_width
    }
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

# main - write input specified input features
st.subheader("User Input Parameters")
st.write(df)

# load iris dataset
iris = datasets.load_iris()

# load classifier from pickle object (generated from another file, hence we don't have to rebuild model every time)
clf = pickle.load(open("iris_classifier.pkl","rb"))

# use classifier to make prediction and determine how likely each type is
prediction = clf.predict(df)
prediction_probabilities = pd.DataFrame(clf.predict_proba(df)).rename(columns=prediction_label,index={0:"Probability"})

# display labels and corresponding probabilities
st.subheader("Class labels and corresponding index number")
st.write(pd.DataFrame(iris.target_names).transpose())

st.subheader("Prediction Probabilities")
st.write(prediction_probabilities)

# show prediction and image of predicted type
st.subheader("Prediction")
st.write("Our model predicts: " + prediction_label[prediction[0]])
st.image(prediction_image[prediction[0]], width = 500)



