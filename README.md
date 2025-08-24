# NexusNews - Real-Time News Intelligence Chatbot 🧠

**🚀 Live Demo:** [**https://huggingface.co/spaces/AbhayR1108/NexusNews-Chatbot**](https://huggingface.co/spaces/AbhayR1108/NexusNews-Chatbot)

---

NexusNews is a sophisticated, AI-powered chatbot built with a Retrieval-Augmented Generation (RAG) architecture. It fetches real-time news articles on any given topic, processes the content, and allows users to ask specific, nuanced questions. The entire pipeline is built using open-source models, making it a powerful and cost-effective solution for on-demand information retrieval.

This project was developed as a portfolio piece to showcase skills in AI, data engineering, and Python development.

## 🌟 Key Features
* **Dynamic Topic Fetching**: Users can specify any news topic to build a knowledge base on the fly.
* **Interactive Web UI**: A clean user interface built with Streamlit allows for easy interaction.
* **Headline View**: Users can expand a section to view all fetched headlines with links to the original sources.
* **100% Open-Source AI**: The entire RAG pipeline is powered by free, high-performance models from Hugging Face, requiring no paid API keys for operation.
* **Source Attribution**: Every answer is backed by the source articles used to generate it, ensuring verifiability.

## 🛠️ Tech Stack
* **Core Framework**: LangChain & Streamlit
* **LLMs & Embeddings**: Hugging Face (`google/flan-t5-base`, `all-MiniLM-L6-v2`)
* **Vector Database**: ChromaDB
* **Data Source**: MediaStack News API
* **Deployment**: Hugging Face Spaces

## 🚀 Getting Started

Follow these instructions to get a local copy up and running.

### Prerequisites
* Python 3.9+
* A free API key from [MediaStack](https://mediastack.com/)

### Installation

1.  **Clone the repository:**
    ```sh
    git clone [https://github.com/AbhayR1104/NexusNews.git](https://github.com/AbhayR1104/NexusNews.git)
    cd NexusNews
    ```

2.  **Create a virtual environment:**
    ```sh
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install the dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

4.  **Set up your API Key:**
    Create a file named `.env` in the root directory and add your API key:
    ```
    MEDIASTACK_API_KEY="your_api_key_here"
    ```

### Running the Application

Execute the Streamlit app from your terminal:
```sh
streamlit run Nexus_News.py
