import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI,GoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_core.messages import HumanMessage

from config import config_info,gemini_safety_settings

load_dotenv()


## load the gemini llm models using langchain
def load_gemini_lang_model(config_info):
    return GoogleGenerativeAI(
        model=config_info['lang_gemini_model_name'],
        temperature=config_info["lang_temp"],
        top_k=config_info['lang_top_k'],
        top_p=config_info['lang_top_p'],
        max_output_tokens=config_info['lang_max_num_output_tokens'],
        safety_settings=gemini_safety_settings,
    )

def load_gemini_lang_chat_model(config_info):
    return ChatGoogleGenerativeAI(
        model=config_info['lang_gemini_model_name'],
        temperature=config_info["lang_temp"],
        top_k=config_info['lang_top_k'],
        top_p=config_info['lang_top_p'],
        max_output_tokens=config_info['lang_max_num_output_tokens'],
        safety_settings=gemini_safety_settings,
    )

def load_gemini_vision_model(config_info):
    return ChatGoogleGenerativeAI(
        model=config_info['vision_gemini_model_name'],
        temperature=config_info["vision_temp"],
        top_k=config_info['vision_top_k'],
        top_p=config_info['vision_top_p'],
        max_output_tokens=config_info['vision_max_num_output_tokens'],
    )

def table_summarization_chain(llm):

    prompt_template = """You are an expert in summarizing tables and texts. Provide a meaningful concise \
        summary for the given table: {table}"""
    
    summarize_prompt = ChatPromptTemplate.from_template(
        prompt_template,
    )
    summarize_chain = {"table": lambda x: x} | summarize_prompt | llm | StrOutputParser()
    return summarize_chain

## function to get the image summary
def image_summarize(vision_llm):

    def get_summary(img_base64,prompt):
        msg = HumanMessage(
        content=[{
            "type":"text",
            "text": f"{prompt}",
        },
        {
            "type":"image_url",
            "image_url": {
                "url":f"data:image/jpeg;base64,{img_base64}"
            },
        },
        ]
        )

        return vision_llm.invoke([msg]).content

    return get_summary

def generate_answer(prompt,llm,prompt_template,mv_retriever,reranker=None,source_path=None):

    if source_path:
        search_filters = {'filter': {"source": source_path}, 'k':15}
    else:
        search_filters = {'k':15}

    mv_retriever.search_kwargs = search_filters

    retrieved_docs = mv_retriever.get_relevant_documents(
        prompt,
    )

    # retrieved_docs_from_vs = mv_retriever.vectorstore.similarity_search(prompt)
    # print(retrieved_docs_from_vs)
    # print("-"*100)

    retrieved_docs_decoded_texts = [txt.decode("utf-8") for txt in retrieved_docs]

    # print(len(retrieved_docs))
    # print(type(retrieved_docs[0]),len(retrieved_docs))

    if reranker:
        results = list(reranker.rerank(query=prompt, documents=retrieved_docs_decoded_texts, top_n=5, model='rerank-english-v2.0'))
    else:
        results = retrieved_docs_decoded_texts
    
    print(results,len(results))

    final_prompt = prompt_template.format(context=results,question=prompt)
    response = llm.invoke(final_prompt).content
    return response
    

if __name__=="__main__":
    pass
    # lang_llm = load_gemini_lang_chat_model(config_info)
    # print(lang_llm.invoke("explain me how we can extract data from the webpage").content)

    # for chunk in lang_llm.stream("explain me how we can extract data from the webpage"):
    #     print(chunk)
    # vision_llm = load_gemini_vision_model(
    #     config_info=config_info,
    # )
    # summarizer = image_summarize(vision_llm)
    # print(summarizer())