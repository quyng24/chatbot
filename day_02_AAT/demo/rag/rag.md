# Lesson 02 - RAG: Retrieval-Augmented Generation

Welcome to **Day 2** of the LangChain Agent Development course!

Yesterday you learned how to craft prompts. Today you will give your agent access to **authoritative external knowledge**.

In this lab you will:

1. **Part 1 - Embeddings**: Convert text into vectors and find semantically similar content
2. **Part 2 - Vector Store**: Store and search large amounts of text data efficiently
3. **Part 3 - RAG Pipeline**: Combine retrieval + LLM to answer questions with real context

## Setup

Make sure Ollama is running and required models is available.

Embedding model: `qwen3-embedding-0.6b` (lightweight embedding model)

Large language model: `gpt-oss:20b-cloud`

`gpt-oss:20b-cloud` is running on Ollama cloud server so we don't need to pull it locally.

Then install the extra packages needed:

```python
%pip install langchain langchain-ollama langchain-community faiss-cpu
```

or
```bash
uv add langchain langchain-ollama langchain-community faiss-cpu
```

## Part 1 - Embeddings

An **embedding** is a list of numbers (a vector) that captures the *meaning* of a piece of text. Two sentences with similar meanings will have vectors that are close together in high-dimensional space - even if they share no words.

```python
from langchain_ollama import OllamaEmbeddings
from config import EMBEDDING_ENDPOINT, EMBEDDING_MODEL, EMBEDDING_API_KEY

# Setup embedding model
embed = OllamaEmbeddings(
    base_url=EMBEDDING_ENDPOINT,
    model=EMBEDDING_MODEL,
    dimensions=64,
    client_kwargs={
        "headers": {'Authorization': 'Bearer ' + EMBEDDING_API_KEY} if EMBEDDING_API_KEY else None
    }
)

print("embedding model:", embed.model)

embeddings = embed.embed_documents([
  "Retrieval-Augmented Generation (RAG) is the process of optimizing the output of a large language model."
])

print("embedding:", embeddings[0][:5], "...")
```

Cosine similarity between 2 vectors:

```python
from utils import cosine_sim

embeddings = [
    [0.1, 0.2],
    [0.1, 0.3],
    [0.4, 0.8]
]

# Question: Which similarity is higher?
print(f"#0 & #1 similarity : {cosine_sim(embeddings[0], embeddings[1]):.4f}")
print(f"#0 & #2 similarity: {cosine_sim(embeddings[0], embeddings[2]):.4f}")
```

### Why embeddings matter for agents

When a user asks a question, we convert it to a vector and search a database for the most *semantically similar* chunks of text. This is called **semantic search** and it is the backbone of Retrieval-Augmented Generation (RAG).

Without embeddings, we would have to rely on keyword matching - which fails for paraphrases, synonyms, and cross-language queries.

```python
database = [
    "Dog has 4 legs.",
    "Cat has 2 forelimbs and 2 hind limbs.",
    "Bird has wings",
]

query = "Which animals have 4 legs?"

embeddings = embed.embed_documents([query, *database])

print("embedding length:", len(embeddings[0]), "\n")

print("query embedding:", embeddings[0][:5], "...")
print("data embeddings:\n", *[f"{e[:5]}...\n" for e in embeddings[1:4]], "\n")

print(f"Query & #1 similarity: {cosine_sim(embeddings[0], embeddings[1]):.4f}")
print(f"Query & #2 similarity: {cosine_sim(embeddings[0], embeddings[2]):.4f}")
print(f"Query & #3 similarity: {cosine_sim(embeddings[0], embeddings[3]):.4f}")
```

## Part 2 - Vector Store

### Overview

A **vector store** is a database designed to store and retrieve vectors (embeddings). Instead of searching for exact keyword matches, it finds semantically similar content by comparing vectors.

**Why vector store matters:**
- Fast retrieval: Search through millions of vectors in milliseconds
- Semantic understanding: Find relevant content even if keywords don't match exactly
- Memory efficient: Store and compare numerical vectors at scale

### Vector Store Concepts

A vector store has two core operations:

1. **Store**: Save pre-computed embeddings into the database
2. **Search**: Find the most similar vectors to a query embedding using distance metrics (e.g., cosine similarity)

For this lab, we'll use **FAISS** (Facebook AI Similarity Search) - a library for fast nearest-neighbor search over dense vectors. FAISS provides a local, in-memory vector store with no infrastructure setup needed.

### Embedding data, save to vector store and semantic search

We'll now create a simple vector store from our knowledge base and perform semantic search.

**Steps:**
1. Create embeddings from documents using the embedding model
2. Save embeddings to FAISS vector store
3. Search the vector store with a query

Let's see how the vector store retrieves the most relevant documents:

```python
from langchain_core.documents import Document
from config import EMBEDDING_ENDPOINT, EMBEDDING_MODEL, EMBEDDING_API_KEY
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings

# Setup embedding model
embed = OllamaEmbeddings(
    base_url=EMBEDDING_ENDPOINT,
    model=EMBEDDING_MODEL,
    dimensions=384,
    client_kwargs={
        "headers": {'Authorization': 'Bearer ' + EMBEDDING_API_KEY} if EMBEDDING_API_KEY else None
    }
)


# -----  Knowledge base -----
docs = [
    Document(page_content="Python is a general-purpose language known for simplicity and readability. It's widely used in data science, automation, and web development.", metadata={"source": "python"}),
    Document(page_content="JavaScript is the primary language for web browsers. It enables interactive user interfaces and is essential for front-end web development.", metadata={"source": "javascript"}),
    Document(page_content="Java is a strongly-typed language known for reliability and scalability. It's commonly used in large-scale enterprise applications and backend systems.", metadata={"source": "java"}),
    Document(page_content="Go is designed for fast execution and concurrent programming. It's popular for building high-performance web servers and cloud applications.", metadata={"source": "go"}),
    Document(page_content="Ruby emphasizes developer productivity and readable code. It's often used with the Rails framework for rapid web development.", metadata={"source": "ruby"}),
    Document(page_content="C++ offers high performance and direct memory control. It's preferred for system programming, game engines, and performance-critical applications.", metadata={"source": "cpp"}),
]

vector_store = FAISS.from_documents(docs, embed)

print(f"Vector store built with {len(docs)} documents.")
```

***Note***: `FAISS.from_documents()` handles two steps internally:

(1) embedding each chunk using the embedding model
(2) storing the vectors in the FAISS index.

In production, you would separate these steps when using other vector databases like Pinecone, Weaviate, or Milvus.

Finally, implement semantic search with embedding vectors:

**Note:** FAISS offers two similar search methods — `similarity_search_with_score` returns a raw distance (lower = closer), while `similarity_search_with_relevance_scores` (used below) returns a normalized relevance score from 0 to 1 (higher = more relevant). Don't confuse the two when reading scores.

```python
# ----- Semantic search -----
query = "I want to build a simple website, which programming language should I use?"

# query = "I want to build a simple landing page, which programming language should I use?"

results = vector_store.similarity_search_with_relevance_scores(query, k=3)

print(f"Query: {query}\n")
for i, (doc, score) in enumerate(results, 1):
    print(f"Chunk {i} (source: {doc.metadata['source']}, score: {score:.4f})")
    print(doc.page_content)
```

### Chunking file, chunk embeddings and semantic search.

In real applications, you often need to search through entire documents. We'll load a document file, split it into chunks, embed those chunks, and perform semantic search.

**Steps:**
1. Load a document file (e.g., .txt, .docx, .pdf)
2. Split the document into smaller chunks
3. Embed all chunks and save to vector store
4. Search across all chunks with a query


First, install the document loader library:

```python
%pip install langchain-text-splitters docx2txt
```

or
```bash
uv add langchain-text-splitters docx2txt
```

Now let's load and chunk a document.

**1. Loading the document**

The document loader reads the entire file as a single text block. We'll load a .docx file from the same directory:

```python
from langchain_community.document_loaders import Docx2txtLoader

# Load document
loader = Docx2txtLoader("customer_account_management.docx")
documents = loader.load()

print("total documents:", len(documents), "\n")

document_content = documents[0].page_content
print(f"content length: {len(document_content)} \n\n")
print("page_content:\n\n", document_content[:204], "...")
```

**2. Splitting into chunks**

The document is too long to process as a whole. We split it into smaller chunks with some overlap to preserve context across chunk boundaries.

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    separators=["\n\n", "\n", ".", ",", " ", ""]
)

# Split document into chunks
chunks = text_splitter.split_documents(documents)

print(f"Total chunks: {len(chunks)} \n")
print(f"Chunk 1 content:\n{chunks[0].page_content}\n\n")
print(f"Chunk 2 content:\n{chunks[1].page_content}")
```

**3. Embedding chunks and save to vector store**

Each chunk is converted into an embedding vector. These vectors are then saved to the FAISS vector store.

```python
from config import EMBEDDING_ENDPOINT, EMBEDDING_MODEL, EMBEDDING_API_KEY
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings


embed = OllamaEmbeddings(
    base_url=EMBEDDING_ENDPOINT,
    model=EMBEDDING_MODEL,
    dimensions=768,
    client_kwargs={
        "headers": {'Authorization': 'Bearer ' + EMBEDDING_API_KEY} if EMBEDDING_API_KEY else None
    }
)

vector_store = FAISS.from_documents(chunks, embed)

print(EMBEDDING_MODEL)

print(f"Embed {len(chunks)} chunks.")
```

**4. Semantic search across chunks**

We can now search the vector store with a query. FAISS will find the most relevant chunks based on semantic similarity, not exact keyword matching.

```python
# Search the vector store
query = "What happens when an account is deactivated?"
# query = "Can email be changed after account creation?"
results = vector_store.similarity_search_with_relevance_scores(query, k=3)

print(f"Query: {query}\n")
print("------------------------------------------------------------")
for i, (doc, score) in enumerate(results, 1):
    print(f"\n\n[ --- Chunk {i} (score: {score:.4f}) --- ]\n\n")
    print(doc.page_content)
    print()
```

We've built a semantic search system: loaded documents, split into chunks, embedded them, and retrieved relevant chunks based on query similarity.

However, the vector store only returns raw text chunks. To generate natural language answers, we need to feed these chunks to an LLM.

## Part 3 - RAG with LangChain retrieval chains

A **Retrieval chain** combines the retriever and the LLM into a single, callable object:

1. The user asks a question.
2. The retriever fetches the top-k relevant chunks from vector store.
3. The chunks are injected into a prompt as context.
4. The LLM generates an answer grounded in that context.

The model is instructed to say *"I don't know"* if the answer is not in the retrieved context - this prevents hallucination.

First, let's setup neccessary tools:

```python
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


from config import (
  LLM_ENDPOINT, LLM_MODEL, LLM_API_KEY,
  EMBEDDING_ENDPOINT, EMBEDDING_MODEL, EMBEDDING_API_KEY
)


# Setup LLM
llm = ChatOllama(
    base_url=LLM_ENDPOINT,
    model=LLM_MODEL,
    temperature=0.3,
    client_kwargs={
        "headers": {'Authorization': 'Bearer ' + LLM_API_KEY} if LLM_API_KEY else None
    },
)


# Setup embedding model
embed = OllamaEmbeddings(
    base_url=EMBEDDING_ENDPOINT,
    model=EMBEDDING_MODEL,
    dimensions=1536,
    client_kwargs={
        "headers": {'Authorization': 'Bearer ' + EMBEDDING_API_KEY} if EMBEDDING_API_KEY else None
    }
)


# Load document
loader = Docx2txtLoader("customer_account_management.docx")
documents = loader.load()


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    separators=["\n\n", "\n", ".", ",", " ", ""]
)

# Split document into chunks
chunks = text_splitter.split_documents(documents)

# Chunk metadata
for i, chunk in enumerate(chunks, 1):
  chunk.metadata = {"file": "customer_account_management.docx", "chunk": i}


print("llm model:", llm.model)
print("embed model:", embed.model)
print("Total chunks:", len(chunks))
```

**Note:** Document loaders and text splitters already populate `metadata["source"]` automatically — `Docx2txtLoader` sets it to the file path, and `split_documents()` copies that metadata into every chunk it produces. The `chunk.metadata = {"file": ..., "chunk": ...}` line above intentionally overwrites that default, replacing the built-in `source` field with our own citation format used later in the prompt.

```python
# Embedding and save to vector store
vector_store = FAISS.from_documents(chunks, embed)

# or following to the production best practice
# vector_store = FAISS.from_documents([], embed) # initial a vector_store with no documents
# vector_store.add_documents(chunks) # Add documents and embed them later

print("Embedding successfully.")
```

#### Now implement a standard RAG flow with basic python implementation:

```python
query = "What happens when an account is deactivated?"

# Search the vector store
results = vector_store.similarity_search_with_relevance_scores(query, k=2)

print(f"Query: {query}\n")
print("------------------------------------------------------------")
for doc, score in results:
    print(f"\n\n[ --- file: {doc.metadata["file"]}, chunk: {doc.metadata["chunk"]}, score: {score:.4f} --- ]\n\n")
    print(doc.page_content[:200], "...")
    print()
```

```python
# Feed to context
from langchain_core.documents import Document

def format_docs(docs: list[Document]):
    return "\n\n".join(f"[{str(doc.metadata)}]\n{doc.page_content}" for doc in docs)

docs = [doc for doc, _score in results]
context = format_docs(docs)

prompt = """Use the following context to answer the user's question.
Only use information from the provided context. Do not use any external knowledge or make up information.
If the answer is not in the context, just say "I don't know".
Cite the source document for each piece of information in your answer (e.g. <citation file="my_file.docx" chunk="10" />.

Context:
'''
{context}
'''

Question: {query}

Answer:""".format(
    context=context,
    query=query
)

print(prompt)
```

```python
# LLM response

response = llm.invoke(prompt)

print("LLM response:\n\n", response.content, "\n\n")
print("LLM usage metadata:", response.usage_metadata)
```

_The RAG flow we just built works as expected._

However, as we add more steps - such as query reranking, query routing, or result filtering - the code becomes scattered and hard to maintain.

#### Simplifying with LCEL

LangChain Expression Language (LCEL) lets us chain these components together into a single, readable pipeline using the `|` operator.

```python
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document


# Setup vectore store as retrieval
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 2})

# Setup format documents into texts
def format_docs(docs: list[Document]):
    return "\n\n".join(f"[{str(doc.metadata)}]\n{doc.page_content}" for doc in docs)

# Setup PromptTemplate
prompt = PromptTemplate.from_template("""Use the following context to answer the user's question.
Only use information from the provided context. Do not use any external knowledge or make up information.
If the answer is not in the context, just say "I don't know".
Cite the source document for each piece of information in your answer (e.g. <citation file="my_file.docx" chunk="10" />.

Context:
'''
{context}
'''

Question: {question}

Answer:""")

print(prompt.template)
```

```python
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

output_parser = StrOutputParser()

# Using LCEL chain for simplicity, scalability, observability, ...
qa_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | output_parser
)

print("qa_chain total steps:", len(qa_chain.steps))
print("qa_chain input type:", qa_chain.input_schema.model_json_schema()["type"])
print("qa_chain output type:", qa_chain.output_schema.model_json_schema()["type"])
```

```python
query = "What happens when an account is deactivated?"
# query = "Can email be changed after account creation?"
# query = "Is subscription plan affected by deactivation?"

# LLM response
response = qa_chain.invoke(query)

print("LLM response:\n\n", response, "\n\n")
```

__Advanced__

So how do we know what components we can add to the chain?

LangChain provides the **Runnable** interface - any component that implements Runnable can be chained together using the `|` (pipe) operator like a __step__.

A Runnable has an **Input** and an **Output**. When chaining Runnables, the output of one component must match the input of the next.

There are many types of Runnables, such as:
- `RunnablePassthrough`: passes input through without modification
- `RunnableParallel`: runs multiple components in parallel
- `RunnableLambda`: wraps a custom Python function as a Runnable
- `RunnableSequence`: chains Runnables sequentially (created with `|` operator)

LangChain abstracts away the implementation details of these components, which can make them difficult to understand at first. However, once you understand the Input/Output contract, you can confidently build complex chains by combining Runnables like building blocks.

```python
# Runnable type of each step
for step in qa_chain.steps:
  print(type(step))

print("\n")

# First step Input/Output schema
first_step =  qa_chain.steps[0]
print("First step input type:", first_step.input_schema.model_json_schema())
print("First step output type:", first_step.output_schema.model_json_schema())

# first_step = RunnableParallel({"context": retriever | format_docs, "question": RunnablePassthrough()})
```

## Summary

In this lesson you learned how to:

- **Embeddings**: Convert text into vector representations and measure semantic similarity between vectors.

- **Vector Store**: Load documents, split them into chunks, embed the chunks, and store them in FAISS for efficient semantic search.

- **RAG Pipeline**: Build a complete RAG system using LangChain that retrieves relevant chunks from the vector store and passes them to an LLM to generate grounded, context-aware answers.

You also learned how to use **LCEL** to chain multiple components together into a maintainable pipeline.
