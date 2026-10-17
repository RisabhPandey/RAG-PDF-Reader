from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate


load_dotenv()


embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = Chroma(
    persist_directory="chroma-db",
    embedding_function=embedding_model
)

retriever = vectorstore.as_retriever(
    search_type = 'mmr',
    search_kwargs = {
        "k": 4,
        "fetch_k": 10,
        "lambda_mult": 0.5
    }
)


llm = ChatMistralAI(model_name="open-mistral-7b")


# Promt Template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant that answers questions based on the context provided. If the context does not contain the answer, respond with 'I could not find the answer in the document.'"),
        ("human","{context}\n\nQuestion: {question}\nAnswer:")
    ]
)

print("\n Rag System is Created\n")
print("Press 0 to exit\n")

while True:
    query = input("You: ")
    if query == "0":
        print("Exiting...")
        break

    docs = retriever.invoke(query)

    context = "\n\n".join([doc.page_content for doc in docs])

    final_prompt = prompt.invoke({
        "context": context,
        "question": query
    })

    response = llm.invoke(final_prompt)

    print(f"\nAssistant: {response.content}\n")


