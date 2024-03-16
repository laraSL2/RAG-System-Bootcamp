import os
from dotenv import load_dotenv

from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings

import streamlit as st
import cohere

from helpers import load_gemini_lang_chat_model, table_summarization_chain, load_gemini_vision_model
from config import config_info

from file_handling import encode_image, get_character_splitter
from config import config_info
from helpers import load_gemini_lang_model,load_gemini_lang_chat_model, table_summarization_chain, load_gemini_vision_model
from helpers import image_summarize, generate_answer


from vdb_handling import create_or_get_vector_retriever, add_document_to_vdb


load_dotenv()

## loading necessary predefined models and objects

embedding_function = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        task_type="retrieval_query",
    )


# doc_path = "/home/lara/Downloads/kalman_filter.pdf"

lang_llm = load_gemini_lang_model(config_info) #load_gemini_lang_chat_model(config_info)
sum_chain = table_summarization_chain(llm=lang_llm)
vision_llm = load_gemini_vision_model(config_info=config_info)
reranker = cohere.Client()

img_summarizer = image_summarize(
            vision_llm=vision_llm,
    )


character_splitter = get_character_splitter()

# prompt to generate final answer for questions
prompt_template = """You are an expert in programmer who has better knowledge in mathematics. Answer based on the provided context. \ 
You must think step by step before answering the question. Reasoning is important. However please do not make up answers.

Context: {context}\n
Question: {question}\n
Answer: """


def main():

    
    st.set_page_config(page_title="PDF Chat bot")
    # st.title("ChatGPT-like clone")
    st.header("QA Bot")

    # client = load_gemini_lang_chat_model(
    #     config_info=config_info,
    # )

    # if "openai_model" not in st.session_state:
    #     st.session_state["openai_model"] = "gpt-3.5-turbo"


    with st.sidebar:
        st.subheader("Create or Load the Database")
        st.session_state.collection_name = st.text_input("Collection name: ",value="test_collection")
        st.session_state.file_dir = st.text_input("Enter directory path: ",value="temp_dir")

        if st.button("Get database ->"):
            print(st.session_state.collection_name)

            st.session_state.mv_retriever = create_or_get_vector_retriever(
                collection_name = st.session_state.collection_name,
                vdb_path = "vector_database",
                local_file_store_path = "local_file_store",
                embedding_function=embedding_function,
            )

        st.subheader("Upload files here")
        pdf_docs = st.file_uploader("Upload files here",accept_multiple_files=True,type=["pdf"])
        print(pdf_docs)
        if st.button("Add to database ->"):
            with st.spinner("Processing"):
                for doc_path in pdf_docs:
                    
                    file_path = os.path.join(st.session_state.file_dir, doc_path.name)

                    add_document_to_vdb(
                            file_path,
                            retriever=st.session_state.mv_retriever,
                            config_info=config_info,
                            sum_chain = sum_chain,
                            character_splitter=character_splitter,
                            id_key = "doc_id",
                            img_encoder=encode_image,
                            img_summarizer=img_summarizer,
                            prompt=None)


    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_input := st.chat_input("What is up?"):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            # stream = client.chat.completions.create(
            #     model=st.session_state["openai_model"],
            #     messages=[
            #         {"role": m["role"], "content": m["content"]}
            #         for m in st.session_state.messages
            #     ],
            #     stream=True,
            # )
            # response = st.write_stream(stream)


            try:
                response = generate_answer(
                    prompt=user_input,
                    llm = lang_llm,
                    prompt_template=prompt_template,
                    mv_retriever=st.session_state.mv_retriever,
                    reranker=reranker,
                    source_path=None,
                )
            except Exception as ex:
                print(f"Issue encountered: {ex}")
                response = "Oops!! something went wrong. can you please ask the question again!!!"

            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})




if __name__=="__main__":
    main()

