import streamlit as st
import os
import requests
from dotenv import load_dotenv

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

st.set_page_config(page_title="NexusNews")


def load_rag_pipeline(topic, article_limit):
    """
    Loads and configures the entire RAG pipeline.
    """
    st.write(f"Fetching {article_limit} news articles for topic: {topic}...")
    
    api_key = os.getenv('MEDIASTACK_API_KEY')
    if not api_key:
        st.error("Error: MEDIASTACK_API_KEY not found in .env file.")
        return None, None

    url = f"http://api.mediastack.com/v1/news?access_key={api_key}&keywords={topic}&languages=en&limit={article_limit}"
    
    response = requests.get(url)
    response_data = response.json()
    
    articles = []
    if 'data' in response_data and response_data.get('data'):
        for item in response_data['data']:
            if item.get('description') and len(item['description']) > 50:
                articles.append({
                    "title": item['title'],
                    "content": item['description'],
                    "url": item['url']
                })
    st.write(f"Successfully fetched {len(articles)} articles.")
    
    if not articles:
        st.warning("No articles found for this topic.")
        return None, None

    docs = [Document(page_content=article['content'], metadata={'source': article['url']}) for article in articles]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunked_docs = text_splitter.split_documents(docs)
    st.write(f"Split {len(docs)} documents into {len(chunked_docs)} chunks.")

    st.write("Creating Knowledge Base (This may take a minute)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(documents=chunked_docs, embedding=embeddings)
    st.success("Knowledge Base created successfully!")

    st.write("Initializing Q&A Model (This may take a few minutes the first time)...")
    model_id = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_length=512)
    llm = HuggingFacePipeline(pipeline=pipe)
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        return_source_documents=True
    )
    st.success("AI is ready to answer questions.")
    
    return qa_chain, articles

# --- Streamlit User Interface ---
# The code below this line runs instantly.

st.title("NexusNews")
st.subheader("Your AI-Powered News Intelligence Chatbot")

load_dotenv()

topic_input = st.text_input("Enter a news topic (e.g., 'Artificial Intelligence', 'Tesla'):", "technology")
limit_input = st.number_input("Number of articles to fetch (max 100):", min_value=10, max_value=100, value=25)

if st.button("Fetch & Analyze News"):
    with st.spinner("Building knowledge base... This will take a few minutes."):
        st.session_state.qa_chain, st.session_state.articles = load_rag_pipeline(topic_input, limit_input)

if 'qa_chain' in st.session_state and st.session_state.qa_chain is not None:
    st.divider()
    st.subheader("Ask a Question")

    with st.expander("View Fetched Headlines"):
        for i, article in enumerate(st.session_state.articles):
            st.write(f"{i + 1}. {article['title']}")
            st.write(f"   [Link]({article['url']})")
    
    user_question = st.text_input("Ask a question about the articles:")

    if user_question:
        with st.spinner("Thinking..."):
            result = st.session_state.qa_chain.invoke({"query": user_question})
            
            st.write("### Answer")
            st.write(result['result'])

            st.write("### Sources")
            sources = {doc.metadata['source'] for doc in result['source_documents']}
            for source in sources:
                st.write(f"- {source}")
