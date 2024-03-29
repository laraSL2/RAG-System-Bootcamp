import os
from pathlib import Path
import uuid

from dotenv import load_dotenv

from langchain_community.vectorstores import Chroma
from langchain.retrievers.multi_vector import MultiVectorRetriever


from langchain.schema.document import Document
from langchain.storage import LocalFileStore
import chromadb


from file_handling import extract_data_from_document, get_img_summary_list

load_dotenv()

## function to create or load the vector database
def create_or_get_vector_database(collection_name,vdb_path,embedding_function):
    persistent_client = chromadb.PersistentClient(
        path=vdb_path,
    )

    collection = persistent_client.get_or_create_collection(
        name=collection_name,
    )

    langchain_chroma = Chroma(
        client=persistent_client,
        collection_name=collection_name,
        embedding_function=embedding_function,
    )

    return langchain_chroma,collection

## function to store raw files - to use with multivector retriever
def create_or_get_local_file_store(local_file_store_path):
    Path(local_file_store_path).mkdir(parents=True,exist_ok=True)
    local_store = LocalFileStore(local_file_store_path)
    return local_store

## create multivector retriever
def create_or_get_vector_retriever(collection_name,vdb_path,local_file_store_path,embedding_function=None,id_key = "doc_id"):
    vdb,_ = create_or_get_vector_database(collection_name,vdb_path,embedding_function)
    local_store = create_or_get_local_file_store(local_file_store_path)

    mv_retriever = MultiVectorRetriever(
        vectorstore=vdb,
        docstore=local_store,
        id_key=id_key,
    )

    return mv_retriever

## add documents to vector database
## preprocessing happens here
def add_document_to_vdb(doc_path,retriever,config_info,sum_chain,character_splitter=None,id_key = "doc_id",img_encoder=None,img_summarizer=None,prompt=None):

    doc_name = doc_path.split(os.path.sep)[-1].split(".")[0].replace(" ","_")
    parent_dir = config_info['image_output_dir_path']
    config_info['image_output_dir_path'] = config_info['image_output_dir_path'] + "/" + doc_name

    texts, tables = extract_data_from_document(doc_path,config_info)

    ## add texts
    if character_splitter:
        chunks = character_splitter.split_text("\n\n".join(texts))
    else:
        chunks = texts
        
    doc_ids = [str(uuid.uuid4()) for _ in range(len(chunks))]
    added_texts = [
        Document(page_content=txt,metadata={id_key:doc_ids[idx],"source":doc_path}) for idx,txt in enumerate(chunks)
    ]
    encoded_chunks = [chunk.encode() for chunk in chunks]

    retriever.vectorstore.add_documents(added_texts)
    retriever.docstore.mset(list(zip(doc_ids,encoded_chunks)))

    ## add tables
    if len(tables):
        table_summaries = sum_chain.batch(tables,{"max_concurrency":10})
        table_ids = [str(uuid.uuid4()) for _ in range(len(table_summaries))]
        summaries_tables = [
            Document(page_content=txt,metadata={id_key:table_ids[idx],"source":doc_path}) for idx,txt in enumerate(table_summaries)
        ]
        encoded_table_summaries = [summary.encode() for summary in table_summaries]
        retriever.vectorstore.add_documents(summaries_tables)
        retriever.docstore.mset(list(zip(table_ids,encoded_table_summaries)))

    #todo: debug this one - sometime works and sometime does not work
    # possible issues - Gemini vision model, function itself as nested one is used
        
    img_summaries_list,img_base64_list = [],[]
    # ## add images 
    # if prompt is None:
    #     prompt = """You are and expert in graph analysis in images and writing comprehensive summaries. You are provided an image and write a comprehensive summary based on the \
    #         provided image. On provide the correct and accurate summary based on the provided image."""
    # img_summaries_list,img_base64_list = get_img_summary_list(
    #     imgs_dir=config_info['image_output_dir_path'],
    #     img_encoder=img_encoder,
    #     image_summarizer=img_summarizer,
    #     prompt=prompt,
    #     )

    # print(img_base64_list[0])
    if len(img_summaries_list) > 0:
        img_ids = [str(uuid.uuid4()) for _ in range(len(img_summaries_list))]
        summaries_imgs = [
            Document(page_content=txt,metadata={id_key:img_ids[idx],"source":doc_path}) for idx,txt in enumerate(img_summaries_list)
        ]
        encoded_imgs = [raw_img.encode() for raw_img in img_base64_list]
        retriever.vectorstore.add_documents(summaries_imgs)
        retriever.docstore.mset(list(zip(img_ids,encoded_imgs)))

    config_info['image_output_dir_path'] = parent_dir
    
