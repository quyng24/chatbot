# Lab 02 - RAG for CV Screening Agent

## Scenario

Your company needs to screen CVs efficiently - recruiters frequently need to extract information from candidate CVs and evaluate their fit against job descriptions.

Instead of manually reading through each CV, you want a CV Screening Agent to **retrieve relevant information from CVs** before generating its assessment.

This is Retrieval-Augmented Generation (RAG): ground the LLM's evaluation in real candidate data from their CVs, helping recruiters ask questions and determine if candidates match the job requirements.

### What you will build

```
Candidate CVs and Job requirements
            ↓ embed
        FAISS Index
            ↓ semantic search
Question → Retrieve top-k relevant CV chunks → LLM → Answer with context
```

### Exercises at a glance

| # | Topic | Time |
|---|-------|------|
| 1 | Read file & chunking | 8 min |
| 2 | Embedding & save to vector store | 5 min |
| 3 | Semantic search | 7 min |
| 4 | Send message to LLM with context | 5 min |
| 5 | RAG with LCEL | 10 min |

## Setup

```python
%pip install langchain langchain_community langchain-ollama langchain-text-splitters faiss-cpu python-dotenv
```

or
```bash
uv add langchain langchain_community langchain-ollama langchain-text-splitters faiss-cpu python-dotenv
```

```python
from langchain_ollama import ChatOllama, OllamaEmbeddings

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
    dimensions=768,
    client_kwargs={
        "headers": {'Authorization': 'Bearer ' + EMBEDDING_API_KEY} if EMBEDDING_API_KEY else None
    }
)


print("llm model:", llm.model)
print("embed model:", embed.model)
```

## Candidate Evaluation Matrix

This evaluation matrix helps assess how well each candidate matches the Backend Developer job description. The matrix compares key requirements from the JD against each candidate's actual experience, skills, and achievements. Use this as a reference to understand the expected outcomes when screening CVs with the RAG agent.

---

### Evaluation Table

| Requirement | Candidate 01 | Candidate 02 | Candidate 03 | Candidate 04 |
|---|---|---|---|---|
| **Job Title & Role** | ✅ Backend Developer | ✅ Backend Developer | ❌ DevOps Engineer (not backend) | ❌ Frontend Developer (not backend) |
| **Experience Level** | ✅ 4 years backend (Exceeds 2+ req) | ✅ 3 years backend (Meets requirement) | ⚠️ 5 years but in DevOps/Infrastructure | ❌ 2 years frontend only |
| **Python/Java/Node.js** | ✅ Proficient in all three | ✅ Java & Python (application level) | ⚠️ Python only for scripting, not development | ❌ JavaScript only (frontend) |
| **PostgreSQL/MySQL** | ✅ Design & optimization | ✅ Design & basic optimization | ⚠️ Administration only (database servers) | ❌ Basic MySQL only |
| **REST API Design** | ✅ Designed 5+ microservices | ✅ Built multiple APIs | ❌ No API development experience | ❌ No backend API experience |
| **Docker & Kubernetes** | ✅ Production deployment & orchestration | ❌ Not mentioned | ✅ Setup & management (infrastructure focus) | ❌ Not mentioned |
| **AWS/Cloud Platform** | ✅ Multiple services (EC2, RDS, Lambda, S3) | ❌ Not mentioned | ✅ Infrastructure setup (not development) | ❌ Not mentioned |
| **Microservices & Architecture** | ✅ Designed scalable systems | ✅ Understanding of APIs | ❌ No architecture design | ❌ No backend architecture |
| **Code Quality & Development** | ✅ Mentored developers, code reviews | ✅ Participates in code reviews | ❌ Infrastructure focus, not coding | ❌ Frontend focus only |

---

### Summary Assessment

| Candidate | Overall Match | Recommendation | Key Strengths | Key Gaps |
|---|---|---|---|---|
| **01** | 🟢 **Excellent (95%)** | **Hire immediately** | Complete backend skill set, proven system design, mentorship experience, all requirements met | None significant |
| **02** | 🟡 **Good (70%)** | **Interview & consider training** | Solid foundation in core backend skills, can grow into role | Missing cloud technologies, intermediate English level |
| **03** | 🔴 **Not Suitable (15%)** | **Do not hire** | Strong infrastructure skills | ❌ Wrong role entirely (DevOps, not Backend). No API/business logic development. Python only for scripting. Database experience is administration, not design. |
| **04** | 🔴 **Not Suitable (10%)** | **Do not hire** | None for backend role | ❌ Wrong specialization (Frontend Developer). No backend development experience. Lacks all required backend skills. |


---

## RAG Workflow


                               ┌──────────────┐
                               │ User Query   │
                               │ "Which       │
                               │ candidates   │
                               │ meet REST    │
                               │ API design?" │
                               └──────┬───────┘
                                      │
                                      ▼
                        ┌────────────────────────┐
                        │ Step 1: Search CV      │
                        │ Vector Store           │
                        │ (Find top-k candidate  │
                        │  CV chunks)            │
                        └────────────┬───────────┘
                                     │
                                     ▼
                     ┌──────────────────────────────┐
                     │ CV Chunks Found:             │
                     │ - Candidate 01: "Designed    │
                     │   5+ microservices..."       │
                     │ - Candidate 02: "Built       │
                     │   RESTful APIs..."           │
                     │ - Candidate 03: "Deployed    │
                     │   applications..."           │
                     │ - Candidate 04: "Built UI    │
                     │   with React..."             │
                     └─────────────┬────────────────┘
                                   │
                                   ▼
                     ┌──────────────────────────────┐
                     │ Step 2: Create Prompt with   │
                     │ Question, JD + CV Context    │
                     └─────────────┬────────────────┘
                                   │
                                   ▼
                     ┌──────────────────────────────┐
                     │ Step 3: Send to LLM          │
                     └─────────────┬────────────────┘
                                   │
                                   ▼
                     ┌──────────────────────────────┐
                     │ LLM Response:                │
                     │ "Candidate 01 meets the      │
                     │  requirement..."             │
                     └──────────────────────────────┘

---
## Exercise 1 - Read & Chunk CV Data (5 min)

### Objective
Load the candidate CVs and split them into smaller, manageable chunks for retrieval.

**Chunking** breaks each CV into smaller pieces so each section can be embedded and searched independently. This helps find the most relevant candidate information when the recruiter asks questions about specific qualifications or experience.

Each chunk should contain the context about which candidate and section it came from (citation). This allows the RAG agent to retrieve specific information about a candidate's experience, skills, or achievements when evaluating fit against the job description.

```python
import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " ", ""]
)

cvs_folder = "cvs"
cv_chunks = []


## TODO - Load all CV files from the cvs folder and break them into smaller chunks with citation metadata
# --- BEGIN ---

for filename in os.listdir(cvs_folder):
    # file path
    file_path = os.path.join(cvs_folder, filename)

    # load file with utf-8 encoding
    loader = ...

    # read file
    documents = ...

    # split into chunks
    doc_chunks = ...

    # add citation metadata: {"file": <File name>, "chunk": <Chunk index in file>}
    for i, chunk in enumerate(doc_chunks):
        chunk.metadata = {...}

    # add to chunk store
    cv_chunks.extend(doc_chunks)

# --- END ---

print("Total CV chunks:", len(cv_chunks))
```

---
## Exercise 2 - Embedding & Save to Vector Store (5 min)

### Objective
Convert each CV chunk into embeddings and store them in a vector store for fast retrieval.

**Embeddings** convert text into numerical vectors that capture meaning. Similar CV sections (e.g., two candidates with Python experience) will have similar vectors, making it easy to find relevant candidate information when the recruiter asks questions.

A **vector store** is a database that saves these embeddings so you can quickly search for chunks semantically similar to a recruiter's query (e.g., "Who has Docker experience?" or "Which candidate has microservices background?").

You will use an embedding model to convert CV chunks into vectors, then add them to a vector store so they can be searched later during candidate screening.

```python
from langchain_community.vectorstores import FAISS

## TODO: Convert CV chunks into vectors
# --- BEGIN ---

cv_vector_store = ...

# --- END ---

print("Embedding successfully.")
```

---
## Exercise 3 - Semantic Search (7 min)

### Objective
Search the CV vector store to match candidate qualifications with job requirements.

**How it works:**
Take a job requirement query, convert it to an embedding, and search the CV vector store to find candidate CV chunks that match that requirement.

**Example flow:**
- Question: "Which candidates meet the REST API design requirement?"
- → Search `cv_vector_store` → Find: "Designed and developed 5+ microservices using Node.js" (Candidate 01), "Built RESTful APIs using Spring Boot" (Candidate 02), etc.

You will implement the semantic search to retrieve relevant candidate qualifications.

**Note:** FAISS offers two similar search methods — `similarity_search_with_score` returns a raw distance (lower = closer), while `similarity_search_with_relevance_scores` (used in this exercise) returns a normalized relevance score from 0 to 1 (higher = more relevant). Don't confuse the two when reading scores.

```python
from langchain_core.documents import Document

query = "Which candidate has 3 years of experience in design and develop RESTful APIs."


def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(f"[{str(doc.metadata)}]\n{doc.page_content}" for doc in docs)


## TODO: Search CV vector store to find matching candidates
# --- BEGIN ---

# Search CV vector store for candidates with k=4
cv_results = ...

# --- END ---

print(f"Query: {query}\n")
print("=" * 60)

print("\nMatching Candidates:")
print("-" * 60)
for doc, score in cv_results:
    print(f"\n[{doc.metadata}] - score: {score:.4f}\n{doc.page_content}\n")
```

### Discussion

1. Why do we need to chunk CV data into smaller pieces instead of embedding the whole CV at once?

2. What is the purpose of converting text into embeddings, and why can't we just use keyword matching to find candidate information?

3. If a recruiter asks "Which candidates can design and build microservices?", how does the vector store find relevant candidate chunks?

---
## Exercise 4 - Send Message to LLM with Context (5 min)

### Objective
Create a prompt and send retrieved candidate information along with the job description to the LLM for evaluation.

You will read the job description text file and combine it with the retrieved CV chunks as context in a prompt, then send it to the LLM so it can generate an accurate assessment of candidate qualifications.

```python
# Read Job Description content as static context
with open("job_description.txt", "r", encoding="utf-8") as f:
    jd_content = f.read()

# Format retrieved CV chunks as context
cv_context = format_docs([doc for doc, _score in cv_results])

## TODO: Complete the prompt with context and question
# --- BEGIN ---

prompt = f"""Use the following context to evaluate candidates against the job description.
Only use information from the provided context. Do not use any external knowledge or make up information.
If a candidate's qualifications are not mentioned, say "Not mentioned in CV".
Cite the source document for each piece of information in your answer (e.g., <citation file="01_Nguyen_Phong.txt" chunk="0" />).

Job Description:
'''
{...}
'''

Candidate Information:
'''
{...}
'''

Question: {...}

Answer:"""

# --- END ---

print(prompt)
```

```python
## TODO: Call LLM with prompt and get response
# --- BEGIN ---
response = ...
# --- END ---

print("LLM response:\n\n", response.content, "\n\n")
print("LLM usage metadata:", response.usage_metadata)
```

---
## Exercise 5 - RAG with LCEL (10 min)

### Objective
Build a complete RAG pipeline by combining retrieval and LLM using LCEL.

**LCEL (LangChain Expression Language)** helps you connect multiple steps (CV retrieval → prompt → LLM) into a single pipeline instead of calling them manually one by one.

The pipeline will:
- Take a recruiter's question about job requirements
- Retrieve matching candidate information from the CV vector store
- Insert candidate information into a prompt with the job description
- Send the prompt to the LLM
- Return the final evaluation of which candidates match

This makes your CV Screening Agent reusable, cleaner, and easier to maintain.

You will define a chain that connects all components together and run it with different queries to evaluate candidates.

```python
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


## TODO: Build a retrieval pipeline with LCEL
# --- BEGIN ---

# Read Job Description content
with open("job_description.txt", "r", encoding="utf-8") as f:
    jd_content = f.read()

# CV vector store retriever with `similarity` search type and k=4
cv_retriever = ...

prompt_template = ...

output_parser = ...

qa_chain = ...

query = "Which candidates meet the REST API design requirement?"

response = qa_chain.invoke(query)

print("LLM response:\n\n", response, "\n\n")

# --- END ---
```

### Test with other queries

```python
query = "Which candidates meet the Backend technologies (Python/Java/Node.js) requirement?"
# query = "Which candidates can optimize database performance?"

response = ...

print("LLM response:\n\n", response, "\n\n")
```

---
## Summary

In this lab you learned how to:

- **Understand embeddings** and use cosine similarity to measure semantic closeness between candidate qualifications and job requirements.
- **Build a domain-specific knowledge base** from candidate CVs and embed them locally using an embedding model.
- **Create a vector store** for CVs and query it with semantic search to find matching candidates.
- **Wire a RAG chain** that retrieves candidate information, injects it into the LLM prompt along with the job description, and produces accurate candidate evaluations based on actual CV data.

