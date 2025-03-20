import streamlit as st
import pickle
import pandas as pd
import streamlit as st
import plotly.express as px

model = pickle.load(open('spam.pkl','rb'))
cv = pickle.load(open('vectorizer.pkl','rb'))

def main():
# 	st.title("Email Spam Classification Application")
# 	st.write("This is a Machine Learning application to classify emails as spam or ham.")
# 	st.subheader("Classification")
# 	user_input=st.text_area("Enter an email to classify" ,height=150)
# 	if st.button("Classify"):
# 		if user_input:
# 			data=[user_input]
# 			print(data)
# 			vec=cv.transform(data).toarray()
# 			result=model.predict(vec)
# 			if result[0]==0:
# 				st.success("This is Not A Spam Email")
# 			else:
# 				st.error("This is A Spam Email")
# 		else:
# 			st.write("Please enter an email to classify.")
# main()

	st.markdown("""
	    <style>
	        /* Change background color */
	        .stApp {
				
				// background: linear-gradient(#DDA0DD, #800080);
				 background-color:#191970;
				
	        }
			    .title {
	                color: brown !important;
	                text-align: center;
	                font-size: 50px;
	                font-weight: bold;
	            }
	            .subheader {
	                color: #FF1493 !important;
	                font-size: 20px;
	                text-align: center;
	            }
				
				label {
	              color: white !important;
				
				}

	    </style>
	""", unsafe_allow_html=True)
		

# App title and description
	st.markdown('<h1 class="title">Email Spam Classification Application</h1>', unsafe_allow_html=True)
	st.markdown('<h3 class="subheader">This is a Machine Learning application to classify emails as spam or ham.</h3>', unsafe_allow_html=True)
	user_input=st.text_area("Enter an email to classify" ,height=150)

	if st.button("Classify"):
			if user_input:
				data=[user_input]
				print(data)
				vec=cv.transform(data).toarray()
				result=model.predict(vec)
				
				if result[0]==0:
					# st.success("✅This is Not A Spam Email")
					st.markdown('<p style="background-color:#2E8B57; color:white; padding:8px; border-radius:5px;  font-weight:bold;">'
					'✅ This is Not A Spam Email</p>', unsafe_allow_html=True)
				else:
					# st.error(" 🚨 This is A Spam Email")
					st.markdown('<p style="background-color:#DC143C; color:white; padding:8px; border-radius:5px;  font-weight:bold;">'
					'🚨 This is A Spam Email</p>', unsafe_allow_html=True)
			else:
				# st.write("Please enter an email to classify.")
				st.markdown('<h3 style="color:red">Please enter an email to classify.</h3>', unsafe_allow_html=True)

			
			# display probability and prob bar
			spam_prob = model.predict_proba(vec)[0][1]  # Probability of spam
			ham_prob = model.predict_proba(vec)[0][0]  # Probability of not spam

			# percent
			spam_percent = spam_prob * 100
			ham_percent = ham_prob * 100

			st.markdown('<h3 style="color:red;">Spam Probability</h3>', unsafe_allow_html=True)
			st.markdown(f'<h4 style="color:orange;">{spam_percent:.2f}%</h4>', unsafe_allow_html=True)  
			st.progress(int(spam_prob * 100))  # Spam probability bar

			st.markdown('<h3 style="color:green;">Ham Probability</h3>', unsafe_allow_html=True)
			st.markdown(f'<h4 style="color:orange;">{ham_percent:.2f}%</h4>', unsafe_allow_html=True)  
			st.progress(int(ham_prob * 100))  # Ham probability bar



	if st.button("Show Distribution"):
			# Load the data
			df = pd.read_csv("./spam.csv", encoding='ISO-8859-1')

			# Display the dataframe
			st.write("Dataset Preview:")
			st.dataframe(df,use_container_width=True)

			# Assuming the class column is labeled correctly, adjust if necessary
			# Count occurrences of each class (Spam vs Not Spam)
			class_counts = df['class'].value_counts().reset_index()
			class_counts.columns = ['Class', 'Count']

			# Calculate percentage
			class_counts['Percentage'] = (class_counts['Count'] / class_counts['Count'].sum()) * 100

			# Display the percentage
			st.write("Spam vs Not Spam Distribution:")
			st.dataframe(class_counts, use_container_width=True)

			# Create a pie chart
			fig = px.pie(class_counts, values='Percentage', names='Class', title="Spam vs Not Spam Distribution", 
						labels={'Class': 'Message Type'})

			# Show the pie chart
			st.plotly_chart(fig)


main()

