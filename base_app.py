"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st
import joblib,os

# Images
from PIL import Image
import pickle

# Data dependencies
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Vectorizer
news_vectorizer = open("resources/count_vect.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("resources/train.csv")

# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """

	# Creates a main title and subheader on your page -
	# these are static across all pages
	st.title("PRIME CLASSIFIER")
	st.subheader("Climate change tweet classification")

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = ["Home", "View", "Feature Engineering", "Prediction", "About Us", "Contact Us"]
	selection = st.sidebar.radio("Choose Option", options)

	# Building out the "Home" page
	if selection == "Home":
		image = Image.open('resources/imgs/climate-change.jpg')
		st.image(image, caption='Climate Change')

		st.markdown("### Not just an app But solution made handy!")
		st.write("The Prime Classifier App provides data and analytics solutions that enable clients to gain valuable insights from their data, make informed decisions in a timely manner, and consistently stay ahead of the competition.")

	# Building out the "View" page
	if selection == "View":
		st.markdown("### Exploratory Data Analysis (EDA)")
		# You can read a markdown file from supporting resources folder
		st.markdown("This section contains insights on the loaded dataset and its output")

		# Display the unprocessed data
		st.markdown("##### Raw Twitter data")
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			st.write(raw[['sentiment', 'message']]) # will write the df to the page

		option = st.sidebar.selectbox('Select visualization', ('Barplots of common words', 'Word cloud of sentiments'))

		if st.checkbox('Show visualizations'):
			if option == 'Barplots of common words':
				image = Image.open('resources/imgs/most_used.png')
				st.image(image)
			else:
				with st.expander('Pro'):
					image = Image.open('resources/imgs/pro.png')
					st.image(image)
				with st.expander('News'):
					image = Image.open('resources/imgs/news.png')
					st.image(image)
				with st.expander('Neutral'):
					image = Image.open('resources/imgs/neutral.png')
					st.image(image)
				with st.expander('Anti'):
					image = Image.open('resources/imgs/Anti.png')
					st.image(image)

	# Building out the "Feature Engineering" page
	if selection == "Feature Engineering":
		st.markdown("### Feature Engineering")
		# You can read a markdown file from supporting resources folder
		st.markdown("This section contains insights on the features that were added to the data")

		# Display the unprocessed data
		st.markdown("##### Balancing of data")
		if st.checkbox('Show unbalanced data'): # data is hidden if box is unchecked
			image = Image.open('resources/imgs/unbalanced.png')
			image1 = image.resize((800, 700))
			st.image(image1)

		if st.checkbox('Show balanced data'): # data is hidden if box is unchecked
			image = Image.open('resources/imgs/balanced.png')
			image1 = image.resize((800, 700))
			st.image(image1)


	# Building out the predication page
	if selection == "Prediction":
		st.info("Prediction with ML Models")
		option = st.sidebar.selectbox(
            'Select the model from the Dropdown',
            ('Logistic Regression', 'Decision Tree', 'KNeighbors', 'Random Forest'))
		# Creating a text box for user input
		tweet_text = st.text_area("Enter Text","Type Here")

		# model selection options
		if option == 'Logistic Regression':
			model = "resources/lgr.pkl"
		elif option == 'Decision Tree':
			model = "resources/dtc.pkl"
		elif option == 'KNeighbors':
			model = "resources/knc.pkl"
		else:
			model = "resources/rf.pkl"

		if st.button("Classify"):
			# Transforming user input with vectorizer
			vect_text = tweet_cv.transform([tweet_text]).toarray()
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
			predictor = joblib.load(open(os.path.join(model),"rb"))
			prediction = predictor.predict(vect_text)

			word = ''
			if prediction == 0:
				word = '"**Neutral**". The tweet neither supports nor refutes the belief of man-made climate change'
			elif prediction == 1:
				word = '"**Pro**". The tweet supports the belief of man-made climate change'
			elif prediction == 2:
				word = '**News**. The tweet links to factual news about climate change'
			else:
				word = '**Anti**. The tweet do not belief in man-made climate change'

			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
			# more human interpretable.
			st.success("Text Categorized as: {}".format(word))

	# Building out the About Us page
	if selection == "About Us":
		st.info("Prime Tech Consult")
		st.write("Prime Tech Consult provides data and analytics solutions that enable clients to gain valuable insights from their data, make informed decisions in a timely manner, and consistently stay ahead of the competition.")
		
		st.info("Our Vision:")
		st.write("To be the lead Tech Solution Plug")

		st.info("Meet the team")
		Damola = Image.open('resources/imgs/damola.png')
		Damola1 = Damola.resize((150, 155))
		Confidence = Image.open('resources/imgs/confidence.JPG')
		Confidence1 = Confidence.resize((150, 155))
		Akani = Image.open('resources/imgs/akani.PNG')
		Akani1 = Akani.resize((150, 155))
		Seye = Image.open('resources/imgs/seye.png')
		Seye1 = Seye.resize((150, 155))
		Samuel = Image.open('resources/imgs/samuel.png')
		Samuel1 = Samuel.resize((150, 155))

		col1, col2, col3, col4 = st.columns(4)
		with col2:
			st.image(Damola1, width=150, caption="Damola: Team Lead")
		with col3:
			st.image(Confidence1, width=150, caption="Confidence: Technical Lead")
		
		col1, col2, col3 = st.columns(3)
		
		with col1:
<<<<<<< HEAD
			st.image(Akani1, width=150, caption="Akanni: Project Manager")
=======
			st.image(Akani1, width=150, caption="Akani: Project Manager")
>>>>>>> 43cd349be4971028f12cdf7b09c4c1fc9fea421f
		with col2:
			st.image(Seye1, width=150, caption="Seye: Data Scientist")
		
		with col3:
			st.image(Samuel1, width=150, caption="Samuel: Data Scientist")

	# Build the Contact us page
	if selection == "Contact Us":
		image = Image.open('resources/imgs/contactus.PNG')
		st.image(image)
		
		col1, col2 = st.columns(2)
		with col1:
			st.subheader("Contact info")
			st.write("44, Idumota, Lagos")
			st.write("Lagos, Nigeria")
			st.write("Telephone:+234 8038930893")
			st.write("WhatsApp:+234 8038930893")
			st.write("Email: info@primetech.com")
			
		with col2:
			st.subheader("Send Us")
			email = st.text_input("Enter your email")
			message = st.text_area("Enter your message")
			st.button("Send")

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
