from langchain_google_genai import (
    HarmBlockThreshold,
    HarmCategory,
)

## some default parameter value settings - mostly used for model loading and data preprocessing

config_info = {}

config_info['doc_max_chars'] = 10000
config_info['doc_new_after_n_chars'] = 9800
config_info['combine_under_n_chars'] = 4000
config_info['image_output_dir_path'] = "extracted_images"


config_info['lang_gemini_model_name'] = 'gemini-pro'
config_info["lang_temp"] = 0.1
config_info['lang_top_k'] = 3
config_info['lang_top_p'] = 0.7
config_info['lang_max_num_output_tokens'] = 750

config_info['vision_gemini_model_name'] = 'gemini-pro-vision'
config_info["vision_temp"] = 0.7
config_info['vision_top_k'] = 1
config_info['vision_top_p'] = 0.7
config_info['vision_max_num_output_tokens'] = 150

gemini_safety_settings = {
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT:HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HARASSMENT:HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH:HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT:HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT:HarmBlockThreshold.BLOCK_NONE,
    # HarmCategory.HARM_CATEGORY_UNSPECIFIED:HarmBlockThreshold.BLOCK_NONE,
}