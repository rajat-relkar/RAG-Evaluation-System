from dotenv import load_dotenv
from .file_reader import extract_paragraph
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

# Runs the model locally via sentence-transformers (downloaded once, no API credits needed).
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = InMemoryVectorStore(embedding)

chunks = extract_paragraph("pdfs/Penguins_ACL.pdf")
vector_store.add_texts(texts=chunks)

print(f"Indexed {len(chunks)} chunks")
