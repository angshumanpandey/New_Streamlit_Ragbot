#!/usr/bin/env python
# coding: utf-8

# In[1]:



# In[3]:


# In[7]:


import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS

# Set your Google API key here
GOOGLE_API_KEY = "AIzaSyDcdj_rMyAJPasM9EX8qPWA7cbnOFro4eM"

# Load the saved vectorstore
# Correct placement of allow_dangerous_deserialization
vectorstore = FAISS.load_local(
    "vectorstore",
    GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GOOGLE_API_KEY,
    ),
    allow_dangerous_deserialization=True
)

# Create a retriever from the vectorstore
retriever = vectorstore.as_retriever()

# Set up the Gemini Pro model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0
)

# Create the RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=False
)

# Streamlit UI
st.set_page_config(page_title="📄 PDF QA with Gemini", layout="wide")
st.title("📄 Ask Questions About Your PDF")

query = st.text_input("Enter your question:")

if query:
    with st.spinner("Searching for the answer..."):
        result = qa_chain.run(query)
        st.success("Answer:")
        st.write(result)


# In[ ]:




