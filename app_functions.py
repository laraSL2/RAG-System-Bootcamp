


### without the UI - used for testing the functionalities

from dotenv import load_dotenv

from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.retrievers.multi_vector import SearchType
from langchain.retrievers.document_compressors import CohereRerank
from langchain.retrievers import ContextualCompressionRetriever

from langchain.schema.document import Document
from langchain.storage import LocalFileStore
import chromadb

import cohere

from file_handling import extract_data_from_document,encode_image, get_character_splitter, get_img_summary_list
from config import config_info
from helpers import load_gemini_lang_chat_model, table_summarization_chain, load_gemini_vision_model
from helpers import image_summarize


from vdb_handling import create_or_get_local_file_store, create_or_get_vector_database,create_or_get_vector_retriever, add_document_to_vdb

load_dotenv()


if __name__=="__main__":

    embedding_function = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        task_type="retrieval_query",
    )

    doc_path = "/home/lara/Downloads/kalman_filter.pdf"

    lang_llm = load_gemini_lang_chat_model(config_info)
    sum_chain = table_summarization_chain(llm=lang_llm)
    vision_llm = load_gemini_vision_model(config_info=config_info)

    character_splitter = get_character_splitter()

    mv_retriever = create_or_get_vector_retriever(
        collection_name = "test_collection",
        vdb_path = "vector_database",
        local_file_store_path = "local_file_store",
        embedding_function=embedding_function,
        )
    
    print(mv_retriever)

    img_summarizer = image_summarize(
        vision_llm=vision_llm,
    )

    # img = encode_image("extracted_images/kalman_filter/figure-5-2.jpg")
    # print(img)
    # print(type(img))
    # print(img_summarizer(img,"summarize the image"))
    print("-"*100)

    print("start processing and adding documents.")
    add_document_to_vdb(doc_path,
                        retriever=mv_retriever,
                        config_info=config_info,
                        sum_chain = sum_chain,
                        character_splitter=character_splitter,
                        id_key = "doc_id",
                        img_encoder=encode_image,
                        img_summarizer=img_summarizer,
                        prompt=None)
    

    # compressor = CohereRerank()
    # compression_retriever = ContextualCompressionRetriever(
    #     base_compressor=compressor, base_retriever=mv_retriever,
    # )

    co = cohere.Client()

    query = "what is Kalman filter and give me a discription of where we can use it."


    mv_retriever.search_type = SearchType.mmr
    source_path = doc_path
    if source_path:
        search_filters = {'filter': {"source": source_path}, 'k':25}
    else:
        search_filters = {'k':25}

    mv_retriever.search_kwargs = search_filters

    retrieved_docs = mv_retriever.get_relevant_documents(
        query,
    )

    retrieved_docs_from_vs = mv_retriever.vectorstore.similarity_search(query)
    print(retrieved_docs_from_vs)
    print("-"*100)

    retrieved_docs_decoded_texts = [txt.decode("utf-8") for txt in retrieved_docs]

    # print(len(retrieved_docs))
    print(type(retrieved_docs[0]),len(retrieved_docs))

    results = list(co.rerank(query=query, documents=retrieved_docs_decoded_texts, top_n=5, model='rerank-english-v2.0'))
    print(results,len(results))