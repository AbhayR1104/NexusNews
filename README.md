# NexusNews - Real-Time News Intelligence Chatbot 🧠

NexusNews is a sophisticated, AI-powered chatbot built with a Retrieval-Augmented Generation (RAG) architecture. It fetches real-time news articles on any given topic, processes the content, and allows users to ask specific, nuanced questions. The entire pipeline is built using open-source models, making it a powerful and cost-effective solution for on-demand information retrieval.

This project was developed as a portfolio piece to showcase skills in AI, data engineering, and Python development.

## 🌟 Key Features
* **Dynamic Topic Fetching**: Users can specify any news topic (e.g., "technology", "sports", "finance") to build a knowledge base on the fly.
* **Interactive Q&A**: An interactive command-line interface allows users to have a conversation with the AI, asking follow-up questions about the fetched articles.
* **Headline View**: Users can request a list of all fetched headlines with links to the original sources.
* **100% Open-Source AI**: The entire RAG pipeline, from embeddings to answer generation, is powered by free, high-performance models from Hugging Face, requiring no paid API keys for operation.
* **Source Attribution**: Every answer is backed by the source articles used to generate it, ensuring verifiability.

## 🛠️ Tech Stack
* **Core Framework**: LangChain
* **LLMs & Embeddings**: Hugging Face (`google/flan-t5-base`, `all-MiniLM-L6-v2`)
* **Vector Database**: ChromaDB
* **Data Source**: MediaStack News API
* **Core Libraries**: Transformers, PyTorch, Requests

## 🏛️ Architecture
The application follows a classic Retrieval-Augmented Generation (RAG) pipeline:

1.  **Data Fetching**: Fetches news articles from the MediaStack API based on a user-defined topic.
2.  **Chunking**: Splits the fetched articles into smaller, manageable chunks using LangChain's `RecursiveCharacterTextSplitter`.
3.  **Embedding & Storage**: Uses a Hugging Face sentence transformer to convert the chunks into vector embeddings, which are then stored in a ChromaDB in-memory vector store.
4.  **Retrieval & Generation**:
    * When a user asks a question, the retriever fetches the most relevant chunks from the vector store.
    * These chunks, along with the user's question, are passed to a generative LLM (Flan-T5), which synthesizes a final, context-aware answer.

## 🚀 Getting Started

Follow these instructions to get a local copy up and running.

### Prerequisites
* Python 3.9+
* A free API key from [MediaStack](https://mediastack.com/)

Execute the main Python script from your terminal to start the interactive session:
```sh
python nexus_news.py
