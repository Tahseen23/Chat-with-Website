import streamlit as st
from langchain_huggingface import HuggingFaceEndpoint
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain.chains.llm import LLMChain
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma
import chromadb
from langchain.load import dumps, loads
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
import os
load_dotenv()

# API configuration
api_key=os.environ.get('api_key')
api = api_key
repo_id = "mistralai/Mistral-7B-Instruct-v0.3"
model = HuggingFaceEndpoint(repo_id=repo_id, huggingfacehub_api_token=api)
repo_id = 'sentence-transformers/all-MiniLM-L6-v2'
embeddings = HuggingFaceInferenceAPIEmbeddings(repo_id=repo_id, api_key=api, add_to_git_credential=True)

# Initialize session state for docs, retriever, and Chroma client
if 'docs' not in st.session_state:
    st.session_state.docs = None
if 'retriever' not in st.session_state:
    st.session_state.retriever = None
if 'chroma_client' not in st.session_state:
    st.session_state.chroma_client = chromadb.Client()

def scrap(link):
    loader = WebBaseLoader(link)
    docs = loader.load()
    return docs

def summarize(docs):
    map_template = """This is the following set of documents{docs}
    Based on the list of docs please identify the main themes
    Helpful Answer:"""
    map_prompt = PromptTemplate.from_template(map_template)
    map_chain = LLMChain(llm=model, prompt=map_prompt)

    reduce_template = """The following is set of summaries:
        {docs}
        Take these and distill it into a final, consolidated summary of the main themes. 
        Helpful Answer:"""
    reduce_prompt = PromptTemplate.from_template(reduce_template)
    reduce_chain = LLMChain(llm=model, prompt=reduce_prompt)

    combine_document_chain = StuffDocumentsChain(
        llm_chain=reduce_chain, document_variable_name='docs'
    )

    reduce_documents_chain = ReduceDocumentsChain(
        combine_documents_chain=combine_document_chain,
        collapse_documents_chain=combine_document_chain,
        token_max=1200,
    )

    map_reduce_chain = MapReduceDocumentsChain(
        llm_chain=map_chain,
        reduce_documents_chain=reduce_documents_chain,
        document_variable_name="docs",
        return_intermediate_steps=False,
    )

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    split_docs = text_splitter.split_documents(docs)
    final = map_reduce_chain.run(split_docs)
    return final

def vector_database(docs_f):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    split_docs = text_splitter.split_documents(docs_f)
    collection = st.session_state.chroma_client.create_collection(name="new_final5")
    vectorestore = Chroma.from_documents(documents=split_docs,
                                         collection_name='new_final5',
                                         embedding=embeddings)
    retriever = vectorestore.as_retriever()
    return retriever

# Main UI layout
st.markdown("<h1 style='text-align: center; color: white;'>Chat with Website</h1>", unsafe_allow_html=True)
st.title("Summarize")

# Summarization form
with st.form('my-form'):
    text = st.text_area("Please provide link of the website")
    submit = st.form_submit_button("Summarize")
    if submit:
        docs = scrap(text)
        if docs:
            st.session_state.docs = docs
            summary = summarize(docs)
            st.info(summary)
            st.session_state.retriever = vector_database(docs)  # Initialize and store the retriever
        else:
            st.error("Failed to scrape the website. Please check the URL and try again.")

# Ask a question related to the summarized article
st.title("Ask a question related to the above article")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Query generation
def generate_multiple_query():
    template = """You are a helpful assistant that generates multiple search queries
    based on the single query related to: {question} \n
    Output (4 queries)"""
    prompt_rag_fusion = ChatPromptTemplate.from_template(template)
    generate_queries = (
        prompt_rag_fusion
        | model
        | StrOutputParser()
        | (lambda x: x.split('\n'))
    )
    return generate_queries

# Reciprocal rank fusion
def reciprocal_rank_fusion(results: list[list], k=60):
    fused_scores = {}
    for docs in results:
        for rank, doc in enumerate(docs):
            doc_str = dumps(doc)
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            previous_score = fused_scores[doc_str]
            fused_scores[doc_str] += 1 / (rank + k)
    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]
    return reranked_results

# Generate output
def generate_output(retrival_chain_rag_fusion, question):
    template = """First classify the question if it is greeting message  then you also greet the user if is not then Answer the following question based on the context:
    {context}
    Question: {question}
    If the question is not related to context and not the greeting then just simply say "Website doesn't have information about your question!" 
    """
    prompt = ChatPromptTemplate.from_template(template)
    final_rag_chain = (
        {"context": retrival_chain_rag_fusion, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )
    result = final_rag_chain.invoke({"question": question})
    return result

# Check for user prompt
if prompt := st.chat_input("What is up?"):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    if st.session_state.docs:
        if st.session_state.retriever is None:
            st.session_state.retriever = vector_database(st.session_state.docs)
        
        with st.chat_message("assistant"):
            query = generate_multiple_query()
            retrieval_chain_rag = query | st.session_state.retriever.map() | reciprocal_rank_fusion
            response = generate_output(retrieval_chain_rag, prompt)
            st.write(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        st.error("No documents available to process. Please provide a valid link and summarize first.")

# Function to clean up Chroma collection
def cleanup_chroma_collection():
    if "chroma_client" in st.session_state:
        try:
            st.session_state.chroma_client.delete_collection("new_final5")
        except Exception as e:
            st.error(f"Error cleaning up Chroma collection: {e}")

# Add a button to end session and clean up
if st.button("End Session"):
    cleanup_chroma_collection()
    st.session_state.docs = None
    st.session_state.retriever = None
    st.session_state.messages = []
    st.success("Session ended and Chroma collection cleaned up.")




