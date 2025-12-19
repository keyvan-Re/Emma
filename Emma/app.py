# -*- coding:utf-8 -*-

import os
import sys
import json
import time
import copy
import shutil
import signal
import logging
import platform

import gradio as gr
import nltk
import torch
import tiktoken

from openai import OpenAI as OpenAIClient
from llama_index.embeddings.openai import OpenAIEmbedding

# from langchain_openai import ChatOpenAI, OpenAI as LangChainOpenAI # (If not used, can be commented out)
from llama_index.llms.openai import OpenAI as LlamaIndexOpenAI
from llama_index.core import Settings
from llama_index.core.indices.prompt_helper import PromptHelper

prompt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
sys.path.append(prompt_path)

# Assuming these imports exist in your local environment
from utils.sys_args import data_args, model_args
from utils.app_modules.utils import *
from utils.app_modules.presets import *
from utils.app_modules.overwrites import *
from utils.prompt_utils import *
from utils.memory_utils import (
    enter_name_llamaindex,
    summarize_memory_event_personality,
    save_local_memory,
    extract_session_summary,
    extract_semantic_memory,
)

# Ensure NLTK data path
nltk.data.path = [os.path.join(os.path.dirname(__file__), "nltk_data")] + nltk.data.path

tokenizer = tiktoken.get_encoding("cl100k_base")

GAPGPT_BASE_URL = os.getenv("GAPGPT_BASE_URL", "https://api.gapgpt.app/v1")
openai_client_cache = {}

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification


LOCAL_MODEL_PATH = r"C:\Users\keyva\MMPL_gpt\Classification Model\final_xlm_r_model_router"

print(f"Loading model from: {LOCAL_MODEL_PATH}")

try:
    _tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH)
    _classifier_model = AutoModelForSequenceClassification.from_pretrained(LOCAL_MODEL_PATH)
    _classifier_model.eval()
    print("Local classifier loaded successfully.")
except Exception as e:
    print(f"Error loading local classifier: {e}")


def get_gapgpt_client(api_key: str) -> OpenAIClient:
    if not api_key:
        raise ValueError("API key is missing while attempting to create a GapGPT client.")
    if api_key not in openai_client_cache:
        openai_client_cache[api_key] = OpenAIClient(api_key=api_key, base_url=GAPGPT_BASE_URL)
    return openai_client_cache[api_key]


os_name = platform.system()
clear_command = 'cls' if os_name == 'Windows' else 'clear'
stop_stream = False


def signal_handler(signal_number, frame):
    global stop_stream
    stop_stream = True


VECTOR_SEARCH_TOP_K = 2

# Update this path to your actual file location
api_path = 'C:\\Users\\keyva\\MMPL_gpt\\api_key_list.txt'


def read_apis(path):
    api_keys_local = []
    if os.path.exists(path):
        with open(path, 'r', encoding='utf8') as f:
            for line in f:
                line = line.strip()
                if line:
                    api_keys_local.append(line)
    return api_keys_local


memory_dir = os.path.expanduser("C:\\Users\\keyva\\MMPL_gpt\\memories\\update_memory_0512_eng.json")

# Ensure directory exists
os.makedirs(os.path.dirname(memory_dir), exist_ok=True)

if not os.path.exists(memory_dir):
    json.dump({}, open(memory_dir, "w", encoding="utf-8"))

global memory
memory = json.load(open(memory_dir, "r", encoding="utf-8"))
language = 'en'
user_keyword = generate_user_keyword()[language]
ai_keyword = generate_ai_keyword()[language]
boot_name = boot_name_dict[language]
boot_actual_name = boot_actual_name_dict[language]
meta_prompt = generate_meta_prompt_dict_chatgpt()[language]
meta_prompt_semantic = generate_meta_prompt_dict_semantic_chatgpt()[language]
meta_prompt_semantic_episodic = generate_meta_prompt_dict_semantic_episodic_chatgpt()[language]
new_user_meta_prompt = generate_new_user_meta_prompt_dict_chatgpt()[language]

api_keys = read_apis(api_path)
# Fallback if file is empty or missing for testing purposes
if not api_keys:
    print(f"Warning: No API keys found in {api_path}.")

new_conversation = False
chatgpt_config = {
    "model": "gpt-4o",
    "temperature": 1,
    "max_tokens": 1024,
    "top_p": 0.95,
    "frequency_penalty": 0.4,
    "presence_penalty": 0.2,
    "n": 1,
}

deactivated_keys = []
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
)


def chatgpt_chat(prompt, system, history, gpt_config, api_index=0):
    """
    Handles the chat request to OpenAI.
    History is expected to be a list of dictionaries: [{'role': 'user', 'content': '...'}, ...]
    """
    retry_times, count = 5, 0
    response = None

    while response is None and count < retry_times:
        try:
            request = copy.deepcopy(gpt_config)
            
            # Initial system and greeting messages
            if data_args.language == 'en':
                message = [
                    {"role": "system", "content": system.strip()},
                    {"role": "user", "content": "Hi!"},
                    {"role": "assistant",
                     "content": f"Hi! I'm {boot_actual_name}! I will give you warm companion!"},
                ]
            else:
                message = [
                    {"role": "system", "content": system.strip()},
                    {"role": "user", "content": "Hi"},
                    {"role": "assistant",
                     "content": f"Hi! I'm {boot_actual_name}! I will give you warm companion!"},
                ]

            # --- FIX: Handle History as List of Dicts ---
            if history:
                # Verify if history is old format (list of lists) or new (list of dicts)
                if isinstance(history[0], list):
                     # Convert old format temporarily if encountered
                     for q, a in history:
                         message.append({"role": "user", "content": str(q)})
                         message.append({"role": "assistant", "content": str(a)})
                elif isinstance(history[0], dict):
                    # New format
                    for msg in history:
                        if msg.get('role') in ['user', 'assistant']:
                            message.append(msg)

            # Add the current prompt
            message.append({"role": "user", "content": f"{prompt}"})

            if not api_keys:
                return "Error: No API Keys available."
                
            client = get_gapgpt_client(api_keys[api_index])
            
            # New OpenAI 1.x Syntax
            response = client.chat.completions.create(messages=message, **request)

        except Exception as e:
            print(f"Chat Error: {e}")
            if 'This key is associated with a deactivated account' in str(e):
                deactivated_keys.append(api_keys[api_index])

            if api_keys:
                api_index = api_index + 1 if api_index < len(api_keys) - 1 else 0
                loop_check = 0
                while api_keys[api_index] in deactivated_keys and loop_check < len(api_keys):
                    api_index = api_index + 1 if api_index < len(api_keys) - 1 else 0
                    loop_check += 1
            count += 1

    if response:
        response = response.choices[0].message.content
    else:
        response = ''
    return response


def classify_query_local(text):
  
    
    # 0: episodic, 1: semantic, 2: semantic_episodic, 3: unrelated
    id2label_map = {
        0: "episodic_memory",
        1: "semantic_memory",
        2: "semantic-episodic",  
        3: "unknown"             
    }

    try:
       
        inputs = _tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=512
        )
        
        
       
        with torch.no_grad():
            outputs = _classifier_model(**inputs)
        
        
        logits = outputs.logits
        probabilities = F.softmax(logits, dim=-1)
        predicted_class_id = torch.argmax(probabilities, dim=-1).item()
        
        
        category = id2label_map.get(predicted_class_id, "unknown")
        
        print(f"Local Classifier: '{text}' -> {category} (Class ID: {predicted_class_id})")
        return category

    except Exception as e:
        print(f"Error in local classification: {e}")
        return "unknown"



def classify_query_openai(text, gpt_config, api_index=0, retry_times=5):
    response = None
    count = 0
    local_deactivated = []
    print("text********:", text)

    system_prompt = """
You are an AI that classifies user queries into one of the following memory types:
- 'episodic_memory': Queries about past personal events, daily life logs, or specific experiences the user has shared (e.g., "What did I eat yesterday?", "Tell me about my trip").
- 'semantic_memory': Queries about facts, preferences, general knowledge the user has taught you, or summaries of their personality (e.g., "What is my favorite color?", "Do I like sci-fi movies?").
- 'semantic-episodic': Complex queries requiring both specific past events and general facts/preferences (e.g., "Based on my food preferences, did I enjoy the dinner last night?").

Output ONLY one of these three strings: 'episodic_memory', 'semantic_memory', or 'semantic-episodic'. If unsure, output 'episodic_memory'.
    """.strip()
    
    if not api_keys:
        return "unknown"

    while response is None and count < retry_times:
        try:
            client = get_gapgpt_client(api_keys[api_index])
            
            # New OpenAI 1.x Syntax
            response = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text.strip()},
                ],
                **copy.deepcopy(gpt_config),
            )
        except Exception as e:
            print(f"Classify Error: {e}")
            if "This key is associated with a deactivated account" in str(e):
                local_deactivated.append(api_keys[api_index])

            api_index = api_index + 1 if api_index < len(api_keys) - 1 else 0
            while api_keys[api_index] in local_deactivated:
                api_index = api_index + 1 if api_index < len(api_keys) - 1 else 0
            count += 1

    if response:
        category = response.choices[0].message.content.strip().lower()
        # Basic cleanup
        if "semantic_memory" in category: category = "semantic_memory"
        elif "semantic-episodic" in category: category = "semantic-episodic"
        elif "episodic_memory" in category: category = "episodic_memory"
        else: category = "unknown"
    else:
        category = "unknown"

    return category


def predict_new(
    text,
    history,
    top_p,
    temperature,
    max_length_tokens,
    max_context_length_tokens,
    user_name,
    user_memory,
    user_memory_index,
    service_context,
    api_index,
    semantic_memory_text,
    query_category,
):
    chatgpt_cfg = {
        "model": "gpt-4o",
        "temperature": temperature,
        "max_tokens": max_length_tokens,
        "top_p": top_p,
        "frequency_penalty": 0.4,
        "presence_penalty": 0.2,
        "n": 1,
    }

    if text == "":
        # Return same history if empty input
        return history, history, "Empty context."
        
    # Ensure history is initialized
    if history is None:
        history = []

    system_prompt, related_memo = build_prompt_with_search_memory_llamaindex(
        history=history,
        query=text, # Changed name to match prompt_utils
        user_memory=user_memory,
        user_name=user_name,
        user_memory_index=user_memory_index,
        service_context=service_context,
        api_keys=api_keys,
        api_index=api_index,
        meta_prompt=meta_prompt,
        new_user_meta_prompt=new_user_meta_prompt,
        data_args=data_args,
        boot_actual_name=boot_actual_name,
        semantic_memory_text=semantic_memory_text,
        query_category=query_category,
        meta_prompt_semantic=meta_prompt_semantic,
        meta_prompt_semantic_episodic=meta_prompt_semantic_episodic,
    )

    # Handle context window slicing manually if needed, 
    # though usually OpenAI manages this, or we slice the list of dicts.
    current_history_for_llm = history
    if len(history) > data_args.max_history * 2: # *2 because 1 user + 1 bot
        current_history_for_llm = history[-(data_args.max_history * 2):]

    response = chatgpt_chat(
        prompt=text,
        system=system_prompt,
        history=current_history_for_llm,
        gpt_config=chatgpt_cfg,
        api_index=api_index,
    )

    torch.cuda.empty_cache()

    # --- FIX: Update History with Dictionaries (Gradio Chatbot format) ---
    new_history = history + [
        {"role": "user", "content": text},
        {"role": "assistant", "content": response}
    ]

    # Save memory logic (Needs to handle dict format, assuming save_local_memory can handle it 
    # OR we convert strictly for saving if your legacy code needs it. 
    # For now, assuming we pass the object as is or convert for save)
    if user_name:
        # If save_local_memory expects list of lists, we might need to adapt it inside that function
        # or pass a converted version. Let's try passing the new format.
        save_local_memory(memory, new_history, user_name, data_args)

    # Return: (Chatbot View, State History, Textbox Reset)
    return new_history, new_history, "Generating..."

# فرض بر این است که توابع و متغیرهای زیر در کد اصلی شما وجود دارند:
# memory, data_args, enter_name_llamaindex, summarize_memory_event_personality,
# extract_session_summary, save_local_memory, classify_query_local, predict_new
# چون در قطعه کد ارسالی نبودند، آن‌ها را ایمپورت یا ماک نمی‌کنم تا کد شما به هم نریزد.

def create_gradio_interface(service_context, api_keys):
    with gr.Blocks(title="EMMA") as demo:
        
        gr.HTML("""
        <style>
            /* کانتینر اصلی وسط‌چین */
            .main-container {
                max-width: 800px !important;
                margin-left: auto !important;
                margin-right: auto !important;
                padding: 20px;
                background-color: #ffffff;
                border-radius: 15px;
                box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            }

            /* استایل دکمه‌های آبی روشن */
            .custom-blue-btn {
                background-color: #E0F7FA !important;
                border: 1px solid #4DD0E1 !important;
                color: #006064 !important;
                font-size: 13px !important;
                font-weight: bold !important;
                border-radius: 8px !important;
                padding: 5px 10px !important;
                transition: 0.3s;
                height: 40px !important;
            }
            .custom-blue-btn:hover {
                background-color: #B2EBF2 !important;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }

            /* === بخش جدید: حذف پس‌زمینه خاکستری === */
            .no-bg {
                background: transparent !important;
                background-color: transparent !important;
                border: none !important;
                box-shadow: none !important;
                padding: 0 !important; /* حذف فاصله‌های اضافی */
                gap: 10px; /* فاصله بین دکمه‌ها */
            }
            /* حذف پس‌زمینه پیش‌فرض Row در نسخه‌های جدید گرادیو */
            .no-bg > .form {
                 background: transparent !important;
                 border: none !important;
            }

            .gr-textbox textarea { font-size: 16px; }
            .gr-chatbot { font-size: 15px; }
                
                            /* مخفی کردن آیکون و متن لودینگ Gradio */
            .gradio-container .main .wrap .status {
                display: none !important;
            }
            .eta-bar {
                display: none !important;
            }
            /* مخفی کردن لودر دایره‌ای یا انیمیشن‌های دیگر */
            .loading {
                display: none !important;
            }
            /* مخفی کردن کانتینر وضعیت در نسخه‌های جدیدتر */
            footer {
                display: none !important;
            }
            /* یک روش کلی‌تر برای مخفی کردن نشانگر وضعیت روی دکمه‌ها یا خروجی‌ها */
            .pending {
                opacity: 1 !important; /* جلوگیری از کم‌رنگ شدن */
            }
            /* مخفی کردن نشانگر نارنجی */
            .progress-text {
                display: none !important;
            }
            .meta-text {
                display: none !important; 
            }
            .loader {
                display: none !important;
            }

        </style>
        """)

        state = gr.State({
            "history": [], 
            "user_name": None,
            "memory": memory,
            "data_args": data_args,
            "service_context": service_context,
            "api_keys": api_keys,
            "api_index": 0,
            "semantic_memory_text": "",
            "new_conversation": True,
            "initialized": False
        })

        with gr.Column(elem_classes=["main-container"]):

            header = gr.Markdown("## 🧠 EMMA: Your Empathetic Mental Health Assistant\nWelcome! Please enter your name to begin.")

            with gr.Accordion("🔐 Start New Session", open=True):
                with gr.Column() as username_row:
                    username_input = gr.Textbox(label="Your Name", placeholder="e.g., Alex")
                    age_input = gr.Textbox(label="Age", placeholder="e.g., 28")
                    gender_input = gr.Dropdown(label="Gender", choices=["Male", "Female", "Other"])
                    occupation_input = gr.Textbox(label="Occupation", placeholder="e.g., Student, Engineer...")
                    residence_input = gr.Textbox(label="Place of Residence", placeholder="e.g., Berlin")
                    submit_name_btn = gr.Button("🎯 Start Session", size="sm", elem_classes=["custom-blue-btn"])

            system_msg = gr.Textbox(label="🔔 System Messages", interactive=False, max_lines=2)

            with gr.Column(visible=False) as chat_interface:
                active_header = gr.Markdown()

                # اینجا Group را برداشتیم چون خودش بوردر و پس‌زمینه می‌اندازد
                chatbot = gr.Chatbot(label="💬 EMMA Conversation", height=500)

                # افزودن کلاس no-bg برای حذف پس‌زمینه خاکستری ردیف ورودی و ارسال
                with gr.Row(elem_classes=["no-bg"]):
                    user_input = gr.Textbox(placeholder="Type your message here...", show_label=False, scale=4, container=False) 
                    # نکته: container=False کادر دور تکست‌باکس را تمیزتر می‌کند
                    submit_btn = gr.Button("📤 Send", size="sm", elem_classes=["custom-blue-btn"], scale=1)

                # افزودن کلاس no-bg برای حذف پس‌زمینه خاکستری ردیف دکمه‌های پایین
                with gr.Row(equal_height=True, elem_classes=["no-bg"]):
                    clear_btn = gr.Button("🧹 Clear", size="sm", elem_classes=["custom-blue-btn"])
                    new_session_btn = gr.Button("🔄 New Session", size="sm", elem_classes=["custom-blue-btn"])
                    switch_user_btn = gr.Button("👥 Switch User", size="sm", elem_classes=["custom-blue-btn"])

        # -------------------------------------------------------
        #   Internal Functions (بدون تغییر منطق، فقط کپی شده)
        # -------------------------------------------------------

        def initialize_session(name, age, gender, occupation, residence, state):
            if not name.strip():
                return (
                    gr.update(visible=True),                 
                    gr.update(visible=False),                
                    "Please enter a valid name.",            
                    gr.update(value=""),                     
                    gr.update(value="## 🧠 EMMA: Your Empathetic Mental Health Assistant\nWelcome! Please enter your name to begin."),
                    state,                                   
                    gr.update(value="")                      
                )

            hello_msg, user_memory, sessions_memory, episodic_memory, semantic_memory = enter_name_llamaindex(
                name, memory, data_args)

            memory[name]["profile"] = {
                "age": age,
                "gender": gender,
                "occupation": occupation,
                "residence": residence
            }

            if name in memory.keys():
                user_memory = summarize_memory_event_personality(data_args, memory, name)

            new_state = state.copy()
            new_state["user_name"] = name
            new_state["memory"] = memory
            new_state["initialized"] = True
            new_state["semantic_memory_text"] = semantic_memory
            new_state["history"] = [] 

            welcome_msg = hello_msg if hello_msg else f"Welcome {name}! How can I help you today?"

            return (
                gr.update(visible=False),                  
                gr.update(visible=True),                   
                welcome_msg,                               
                gr.update(value=f"## 🧠 EMMA: Session for {name}"), 
                gr.update(value=""),                       
                new_state,
                gr.update(value="")                        
            )

        def switch_user(state):
            if state["initialized"] and state["user_name"] in state["memory"]:
                if state["memory"][state["user_name"]]["sessions"]:
                    previous_session = state["memory"][state["user_name"]]["sessions"][-1]
                    summary = extract_session_summary(
                        previous_session["conversation"],
                        previous_session["date"],
                        len(state["memory"][state["user_name"]]["sessions"]) - 1
                    )
                    state["memory"][state["user_name"]]["episodic_memory"].append(summary)

            new_state = {
                "history": [],
                "user_name": None,
                "memory": memory,
                "data_args": data_args,
                "service_context": service_context,
                "api_keys": api_keys,
                "api_index": 0,
                "semantic_memory_text": "",
                "new_conversation": True,
                "initialized": False
            }

            return (
                gr.update(visible=True),                   
                gr.update(visible=False),                  
                "Enter a new username to continue.",       
                gr.update(value=""),                       
                gr.update(value="## 🧠 EMMA: Your Empathetic Mental Health Assistant\nPlease enter your name to begin."),
                new_state,
                gr.update(value=""),                       
                gr.update(value=[])                        
            )

        def respond(message, state):
            if not state["initialized"]:
                return [], state, "Please enter your name first."

            if not message.strip():
                return state["history"], state, "Empty input."

            hello_msg, user_memory, sessions_memory, episodic_memory, semantic_memory = enter_name_llamaindex(
                state["user_name"], memory, data_args)

            memo, semantic_memory_text = save_local_memory(
                memory, state["history"], state["user_name"], data_args
            )

            query_category = classify_query_local(message)

            if query_category == "semantic_memory":
                user_memory_index = semantic_memory
            elif query_category == "episodic_memory":
                user_memory_index = episodic_memory
            elif query_category == "semantic-episodic":
                user_memory_index = episodic_memory
            else:
                user_memory_index = None

            chatbot_view, history, msg = predict_new(
                text=message,
                history=state["history"],
                top_p=0.95,
                temperature=1,
                max_length_tokens=1024,
                max_context_length_tokens=200,
                user_name=state["user_name"],
                user_memory=state["memory"],
                user_memory_index=user_memory_index,
                service_context=state["service_context"],
                api_index=state["api_index"],
                semantic_memory_text=semantic_memory_text,
                query_category=query_category
            )

            new_state = state.copy()
            new_state["history"] = history

            return chatbot_view, new_state, "" 

        def clear_chat(state):
            new_state = state.copy()
            new_state["history"] = []
            return [], new_state, "Conversation cleared."

        def new_session(state):
            if state["user_name"] in state["memory"]:
                if state["memory"][state["user_name"]]["sessions"]:
                    previous = state["memory"][state["user_name"]]["sessions"][-1]
                    summary = extract_session_summary(
                        previous["conversation"],
                        previous["date"],
                        len(state["memory"][state["user_name"]]["sessions"]) - 1
                    )
                    state["memory"][state["user_name"]]["episodic_memory"].append(summary)

            new_s = {
                "session_id": len(state["memory"][state["user_name"]]["sessions"]),
                "date": time.strftime("%Y-%m-%d"),
                "conversation": []
            }
            state["memory"][state["user_name"]]["sessions"].append(new_s)

            new_state = state.copy()
            new_state["history"] = []
            new_state["new_conversation"] = True

            return [], new_state, f"🆕 New session started (ID: {new_s['session_id']})."

        # -------------------------------------------------------
        # Buttons Events
        # -------------------------------------------------------

        submit_name_btn.click(
            initialize_session,
            inputs=[username_input, age_input, gender_input, occupation_input, residence_input, state],
            outputs=[username_row, chat_interface, system_msg, active_header, header, state, username_input]
        )

        submit_btn.click(respond,
            inputs=[user_input, state],
            outputs=[chatbot, state, system_msg])

        user_input.submit(respond,
            inputs=[user_input, state],
            outputs=[chatbot, state, system_msg])

        switch_user_btn.click(
            switch_user,
            inputs=[state],
            outputs=[username_row, chat_interface, system_msg, active_header, header, state, username_input, chatbot]
        )

        clear_btn.click(clear_chat,
            inputs=[state],
            outputs=[chatbot, state, system_msg])

        new_session_btn.click(new_session,
            inputs=[state],
            outputs=[chatbot, state, system_msg]
        )

    return demo


def main():
    """Main function to initialize and launch the interface."""
    gapgpt_api_key = os.getenv("GAPGPT_API_KEY")
    
    if not gapgpt_api_key:
        print("Warning: GAPGPT_API_KEY environment variable is not set. Proceeding with keys from file.")
    else:
        # تغییر مهم ۱: ست کردن متغیر محیطی برای جلوگیری از خطاهای ناگهانی کتابخانه‌های وابسته
        os.environ["OPENAI_API_KEY"] = gapgpt_api_key

    # تنظیم LLM (مدل زبانی)
    llm = LlamaIndexOpenAI(
        model="gpt-4o",
        temperature=1,
        max_tokens=1024,
        top_p=0.95,
        frequency_penalty=0.4,
        presence_penalty=0.2,
        api_key=gapgpt_api_key,
        api_base=GAPGPT_BASE_URL,
    )

    # تغییر مهم ۲: تنظیم مدل Embedding با همان کلید و آدرس GapGPT
    # اگر سرویس دهنده شما از امبدینگ پشتیبانی نمی‌کند، این بخش نیاز به تغییر به مدل لوکال دارد
    embed_model = OpenAIEmbedding(
        api_key=gapgpt_api_key,
        api_base=GAPGPT_BASE_URL,
        model="text-embedding-3-small" # یا هر مدلی که سرویس شما پشتیبانی می‌کند
    )

    # اعمال تنظیمات سراسری
    Settings.llm = llm
    Settings.embed_model = embed_model  # <--- این خط جلوی خطای فعلی را می‌گیرد

    Settings.prompt_helper = PromptHelper(
        context_window=4096,
        num_output=256,
        chunk_overlap_ratio=20 / 4096,
        tokenizer=tokenizer,
    )

    # توجه: مطمئن شوید متغیر api_keys در اینجا تعریف شده باشد یا از args خوانده شود
    # اگر api_keys در کد شما تعریف نشده، احتمالاً باید آن را بارگذاری کنید یا اگر استفاده نمی‌شود حذف کنید.
    # فرض بر این است که api_keys قبلاً در کد شما تعریف شده است:
    if 'api_keys' not in locals():
        api_keys = {} # یا هر مقداری که کد شما انتظار دارد

    demo = create_gradio_interface(Settings, api_keys)
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=True
    )


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    main()
