# Website Summarizer and Chatbot

This project is a web application that provides summaries of websites and allows users to interact with a chatbot for questions related to the website content. The application utilizes the mistralai/Mistral-7B-Instruct-v0.3 model for language processing and the sentence-transformers/all-MiniLM-L6-v2 model for embedding. The application is built using Streamlit and LangChain for a seamless and interactive user experience

![second](<Screenshot (29).png>)

# Features
1. Website Summarization: Enter a website URL to receive a summary of its content.
2. Interactive Chatbot: Ask questions related to the website content and get informative responses.
3. Session Management: End sessions to clean up the vector database, ensuring efficient use of resources.


![first](<Screenshot (26).png>)


# Models Used
* Language Model: mistralai/Mistral-7B-Instruct-v0.3
* Embedding Model: sentence-transformers/all-MiniLM-L6-v2

# Technologies
* Streamlit: For creating the web interface.
* LangChain: For handling language processing and integration with the models.

# Usage
1. Enter Website URL: On the home page, enter the URL of the website you want to summarize.
2. View Summary: The application will display a summary of the website content.
3. Ask Questions: Use the chatbot interface to ask questions related to the website content.
4. End Session: Click the "End Session" button to clean the vector database and free up resources.

![last](<Screenshot (27).png>)