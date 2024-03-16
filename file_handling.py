import os 
import base64

from langchain.text_splitter import RecursiveCharacterTextSplitter

from unstructured.partition.pdf import partition_pdf
from config import config_info
from helpers import load_gemini_lang_chat_model, table_summarization_chain

## use unstructured library to extract texts, tables, images from the pdf documents
def extract_data_from_document(doc_path,config_info):
    pdf_path = doc_path

    raw_pdf_elements = partition_pdf(
        filename=pdf_path,
        strategy="hi_res",
        extract_images_in_pdf=True,
        infer_table_structure=True,
        chunking_strategy='by_title',
        max_characters=config_info['doc_max_chars'],
        new_after_n_chars=config_info['doc_new_after_n_chars'],
        combine_text_under_n_chars = config_info['combine_under_n_chars'],
        extract_image_block_output_dir=config_info['image_output_dir_path']
    )

    tables = []
    texts = []

    for element in raw_pdf_elements:
        if "unstructured.documents.elements.Table" in str(type(element)):
            tables.append(str(element))
        elif "unstructured.documents.elements.CompositeElement" in str(type(element)):
            texts.append(str(element))

    return texts,tables

## images - encoding
def encode_image(image_path):
    # base64 string
    with open(image_path,'rb') as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
    
## chunking mechanism
def get_character_splitter(chunk_size=750,chunk_overlap=150,separators=["\n"]):
    return RecursiveCharacterTextSplitter(
        separators=separators,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    
## preprocessing images to generate summaries
def get_img_summary_list(imgs_dir,img_encoder,image_summarizer,prompt):

    img_base64_list = []
    img_summaries_list = []

    for img_file in sorted(os.listdir(imgs_dir)):
        if img_file.endswith(".jpg") or img_file.endswith(".png"):
            img_path = os.path.join(imgs_dir,img_file)
            base64_image = img_encoder(img_path)
            img_base64_list.append(base64_image)
            img_summaries_list.append(image_summarizer(base64_image,prompt))

    return img_summaries_list,img_base64_list

if __name__=="__main__":

    lang_llm = load_gemini_lang_chat_model(config_info)
    sum_chain = table_summarization_chain(llm=lang_llm)

    doc_path = "/home/lara/Downloads/kalman_filter.pdf"
    texts, tables = extract_data_from_document(doc_path,config_info)

    table_summaries = sum_chain.batch(tables,{"max_concurrency":10})

    print(len(texts))
    print(len(tables))
    print(table_summaries)