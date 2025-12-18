# -*- coding:utf-8 -*-
import os, shutil
import logging
import sys, openai
import copy
import time, platform
import signal, json
import gradio as gr
import nltk
import torch
from langchain.llms import AzureOpenAI, OpenAIChat

prompt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
sys.path.append(prompt_path)
from utils.sys_args import data_args, model_args
from utils.app_modules.utils import *
from utils.app_modules.presets import *
from utils.app_modules.overwrites import *
from utils.prompt_utils import *
from utils.memory_utils import enter_name_llamaindex, summarize_memory_event_personality, save_local_memory, extract_session_summary,extract_semantic_memory

nltk.data.path = [os.path.join(os.path.dirname(__file__), "nltk_data")] + nltk.data.path
from llama_index import LLMPredictor, GPTSimpleVectorIndex, PromptHelper, ServiceContext

os_name = platform.system()
clear_command = 'cls' if os_name == 'Windows' else 'clear'
stop_stream = False

def signal_handler(signal, frame):
    global stop_stream
    stop_stream = True

VECTOR_SEARCH_TOP_K = 2

api_path = 'C:\\Users\\keyva\\MMPL_gpt\\api_key_list.txt'

def read_apis(api_path):
    api_keys = []
    with open(api_path, 'r', encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line:
                api_keys.append(line)
    return api_keys

memory_dir = os.path.expanduser("C:\\Users\\keyva\\MMPL_gpt\\memories\\update_memory_0512_eng.json")

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
new_conversation = False  # Start fresh on first entry
chatgpt_config = {
    "model": "gpt-4o",
    "temperature": 1,
    "max_tokens": 1024,
    "top_p": 0.95,
    "frequency_penalty": 0.4,
    "presence_penalty": 0.2,
    'n': 1
}

deactivated_keys = []
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
)

def chatgpt_chat(prompt, system, history, gpt_config, api_index=0):
    retry_times, count = 5, 0
    response = None
    while response is None and count < retry_times:
        try:
            request = copy.deepcopy(gpt_config)
            
            if data_args.language == 'en':
                message = [
                    {"role": "system", "content": system.strip()},
                    {"role": "user", "content": "Hi!"},
                    {"role": "assistant", "content": f"Hi! I'm {boot_actual_name}! I will give you warm companion!"}]
            else:
                message = [
                    {"role": "system", "content": system.strip()},
                    {"role": "user", "content": "Hi"},
                    {"role": "assistant", "content": f"Hi! I'm{boot_actual_name}！I will give you warm companion!"}]
            
            for query, response in history:
                message.append({"role": "user", "content": query})
                message.append({"role": "assistant", "content": response})
            message.append({"role": "user", "content": f"{prompt}"})
            
            response = openai.ChatCompletion.create(**request, messages=message)
            
        except Exception as e:
            print(e)
            if 'This key is associated with a deactivated account' in str(e):
                deactivated_keys.append(api_keys[api_index])
            api_index = api_index + 1 if api_index < len(api_keys) - 1 else 0
            while api_keys[api_index] in deactivated_keys:
                api_index = api_index + 1 if api_index < len(api_keys) - 1 else 0
            openai.api_key = api_keys[api_index]
            count += 1
    
    if response:
        response = response['choices'][0]['message']['content']
    else:
        response = ''
    return response

def classify_query_openai(text, chatgpt_config, api_index=0, retry_times=5):
    response = None
    count = 0
    deactivated_keys = []  # Track deactivated API keys
    print("text********:",text)
    system_prompt = """
You are an AI that classifies user queries into one of the following memory types:
- 'episodic_memory' if the query asks about **recent experiences, emotions, or specific past discussions** (including references to previously mentioned people, places, or events).
- 'semantic_memory' if the query asks about **long-term traits, personality, or values**.
- 'semantic-episodic' if the query requires a combination of both **personality traits or behavioral patterns** (semantic) and **recent emotional states, experiences, or insights** (episodic).

### Instructions:
- Return ONLY one of these exact labels: "episodic_memory", "semantic_memory", or "semantic-episodic".
- Do NOT include explanations or additional text.
- If a query could belong to multiple categories, return the **most appropriate single label**.

### Clarifications:
- If the query refers to a **specific past discussion**, or a **previously mentioned person, place, or event**, or **recent experience or emotion** → classify as **episodic_memory**.
- If the query is about **enduring traits, personality, or values** → classify as **semantic_memory**.
- If the query involves **how a recent experience relates to personality**, or **how a recurring behavior is influenced by past emotional states**, or anything that **blends traits and emotions** → classify as **semantic-episodic**.

### Examples:
User: "What did we talk about last time?"  
AI: episodic_memory  

User: "How have I been feeling lately?"  
AI: episodic_memory  

User: "Do you know my friend Ali?"  
AI: episodic_memory  

User: "What kind of person am I?"  
AI: semantic_memory  

User: "What are my core values?"  
AI: semantic_memory  

User: "How do I usually react to stress?"  
AI: semantic-episodic  

User: "Was my anxiety affecting how I interacted with others recently?"  
AI: semantic-episodic  

User: "Do my personality traits make me more vulnerable to stress?"  
AI: semantic-episodic  

User: "What did I feel after the job interview and how does that reflect on my usual behavior?"  
AI: semantic-episodic

User: "How has my personality affected my recent relationships?"  
AI: semantic-episodic  

User: "Do I tend to get anxious in work situations based on my personality?"  
AI: semantic-episodic  

User: "How has my openness to experience influenced my recent decisions?"  
AI: semantic-episodic
"""
    print("system_prompt+++++++++++:",system_prompt)
    
    while response is None and count < retry_times:
        try:
            request = copy.deepcopy(chatgpt_config)
            messages = [
                {"role": "system", "content": system_prompt.strip()},
                {"role": "user", "content": text.strip()}
            ]

            response = openai.ChatCompletion.create(
                **request,
                messages=messages,
            )

        except Exception as e:
            print(f"Error: {e}")
            if "This key is associated with a deactivated account" in str(e):
                deactivated_keys.append(api_keys[api_index])
            api_index = api_index + 1 if api_index < len(api_keys) - 1 else 0
            while api_keys[api_index] in deactivated_keys:
                api_index = api_index + 1 if api_index < len(api_keys) - 1 else 0
            openai.api_key = api_keys[api_index]
            count += 1

    if response:
        category = response["choices"][0]["message"]["content"].strip().lower()
        if category not in ["semantic_memory", "episodic_memory", "semantic-episodic"]:

            category = "unknown"
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
    query_category
):
    chatgpt_config = {
        "model": "gpt-4o",
        "temperature": temperature,
        "max_tokens": max_length_tokens,
        "top_p": top_p,
        "frequency_penalty": 0.4,
        "presence_penalty": 0.2,
        'n': 1
    }
    
    #Question_category = classify_query_openai(text, chatgpt_config)
   # print("Question_category:", Question_category)

    if text == "":
        return history, history, "Empty context."
    
    system_prompt, related_memo = build_prompt_with_search_memory_llamaindex(
        history,
        text,
        user_memory,
        user_name,
        user_memory_index,
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
        meta_prompt_semantic_episodic=meta_prompt_semantic_episodic
    )
    
    if len(history) > data_args.max_history:
        history = history[data_args.max_history:]
    
    response = chatgpt_chat(
        prompt=text,
        system=system_prompt,
        history=history,
        gpt_config=chatgpt_config,
        api_index=api_index
    )
    result = response
    
    torch.cuda.empty_cache()
    
    a, b = [[y[0], y[1]] for y in history] + [[text, result]], history + [[text, result]]
    
    if user_name:
        save_local_memory(memory, b, user_name, data_args)
    
    return a, b, "Generating..."

def create_gradio_interface(service_context, api_keys):
    """Creates the Gradio chat interface with user switching capability."""

    with gr.Blocks(title="EMMA", css="""
        .mobile-button button {
            width: 100% !important;
            padding: 12px !important;
            font-size: 16px !important;
            border-radius: 10px !important;
            background: linear-gradient(to right, #ff9966, #ff5e62) !important;
            color: white !important;
            font-weight: bold;
        }
        .gr-button-primary {
            background-color: #6a11cb !important;
            background-image: linear-gradient(to right, #6a11cb, #2575fc) !important;
            color: #fff !important;
            font-weight: bold !important;
            border: none !important;
        }
        .gr-button-secondary {
            background-color: #f7971e !important;
            background-image: linear-gradient(to right, #f7971e, #ffd200) !important;
            color: #000 !important;
            font-weight: bold !important;
            border: none !important;
        }
        .gr-textbox textarea {
            font-size: 16px;
        }
        .gr-chatbot {
            font-size: 15px;
        }
    """) as demo:

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

        header = gr.Markdown("## 🧠 EMMA: Your Empathetic Mental Health Assistant\nWelcome! Please enter your name to begin.")

        # User profile section
        with gr.Accordion("🔐 Start New Session", open=True):
            with gr.Column() as username_row:
                username_input = gr.Textbox(label="Your Name", placeholder="e.g., Alex")
                age_input = gr.Textbox(label="Age", placeholder="e.g., 28")
                gender_input = gr.Dropdown(label="Gender", choices=["Male", "Female", "Other"])
                occupation_input = gr.Textbox(label="Occupation", placeholder="e.g., Student, Engineer...")
                residence_input = gr.Textbox(label="Place of Residence", placeholder="e.g., Berlin")
                submit_name_btn = gr.Button("🎯 Start Session", variant="primary")

        system_msg = gr.Textbox(label="🔔 System Messages", interactive=False, max_lines=2)

        with gr.Column(visible=False) as chat_interface:
            active_header = gr.Markdown()

            with gr.Group():
                chatbot = gr.Chatbot(label="💬 EMMA Conversation")

                with gr.Row():
                    user_input = gr.Textbox(placeholder="Type your message here...", show_label=False)
                    submit_btn = gr.Button("📤 Send", variant="primary", elem_classes=["mobile-button"])

                with gr.Row(equal_height=True):
                    clear_btn = gr.Button("🧹 Clear", variant="secondary", elem_classes=["mobile-button"])
                    new_session_btn = gr.Button("🔄 New Session", variant="secondary", elem_classes=["mobile-button"])
                    switch_user_btn = gr.Button("👥 Switch User", variant="primary", elem_classes=["mobile-button"])

        # --- Internal Functions ---

        def initialize_session(name, age, gender, occupation, residence, state):
            if not name.strip():
                return (
                    gr.update(visible=True),
                    gr.update(visible=False),
                    "Please enter a valid name.",
                    gr.update(),
                    gr.update(visible=True),
                    state,
                    gr.update()
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

            welcome_msg = hello_msg if hello_msg else f"Welcome {name}! How can I help you today?"

            return (
                gr.update(visible=False),
                gr.update(visible=True),
                welcome_msg,
                gr.update(value=f"## 🧠 EMMA: Session for {name}"),
                gr.update(visible=False),
                new_state,
                gr.update(value="")
            )

        def switch_user(state):
            if state["initialized"] and state["user_name"] in state["memory"]:
                if state["memory"][state["user_name"]]["sessions"]:
                    previous_session = state["memory"][state["user_name"]]["sessions"][-1]
                    session_summary = extract_session_summary(
                        previous_session["conversation"],
                        previous_session["date"],
                        len(state["memory"][state["user_name"]]["sessions"]) - 1
                    )
                    state["memory"][state["user_name"]]["episodic_memory"].append(session_summary)

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
                "Enter a new username to continue",
                gr.update(),
                gr.update(visible=True, value="## 🧠 EMMA: Your Empathetic Mental Health Assistant\nPlease enter your name to begin."),
                new_state,
                gr.update(value=""),
                []
            )

        def respond(message, state):
            if not state["initialized"]:
                return state["history"], state, "Please enter your name first.", state
            if not message.strip():
                return state["history"], state, "Empty input.", state

            hello_msg, user_memory, sessions_memory, episodic_memory, semantic_memory = enter_name_llamaindex(
                state["user_name"], memory, data_args)
            memo, semantic_memory_text = save_local_memory(memory, state["history"], state["user_name"], data_args)

            query_category = classify_query_openai(message, chatgpt_config)

            if query_category == "semantic_memory":
                user_memory_index = semantic_memory
            elif query_category == "episodic_memory":
                user_memory_index = episodic_memory
            elif query_category == "semantic-episodic":
                user_memory_index = episodic_memory  # or combine both
            else:
                user_memory_index = None

            history_state, history, msg = predict_new(
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
            return history_state, new_state, msg, new_state

        def clear_chat(state):
            new_state = state.copy()
            new_state["history"] = []
            return [], new_state, "Conversation cleared.", new_state

        def new_session(state):
            if state["user_name"] in state["memory"]:
                if state["memory"][state["user_name"]]["sessions"]:
                    previous_session = state["memory"][state["user_name"]]["sessions"][-1]
                    session_summary = extract_session_summary(
                        previous_session["conversation"],
                        previous_session["date"],
                        len(state["memory"][state["user_name"]]["sessions"]) - 1
                    )
                    state["memory"][state["user_name"]]["episodic_memory"].append(session_summary)

                    state["memory"][state["user_name"]]["semantic_memory"] = extract_semantic_memory(
                        session_summary, state["memory"][state["user_name"]]["episodic_memory"]
                    )

            new_session = {
                "session_id": len(state["memory"][state["user_name"]]["sessions"]),
                "date": time.strftime("%Y-%m-%d"),
                "conversation": []
            }
            state["memory"][state["user_name"]]["sessions"].append(new_session)

            new_state = state.copy()
            new_state["history"] = []
            new_state["new_conversation"] = True
            return [], new_state, f"🆕 New session started (ID: {new_session['session_id']}).", new_state

        # --- Button Bindings ---
        submit_name_btn.click(
            initialize_session,
            inputs=[username_input, age_input, gender_input, occupation_input, residence_input, state],
            outputs=[username_row, chat_interface, system_msg, active_header, header, state, username_input]
        )

        submit_btn.click(
            respond,
            inputs=[user_input, state],
            outputs=[chatbot, state, system_msg, state]
        )

        user_input.submit(
            respond,
            inputs=[user_input, state],
            outputs=[chatbot, state, system_msg, state]
        )

        switch_user_btn.click(
            switch_user,
            inputs=[state],
            outputs=[username_row, chat_interface, system_msg, active_header, header, state, username_input, chatbot]
        )

        clear_btn.click(
            clear_chat,
            inputs=[state],
            outputs=[chatbot, state, system_msg, state]
        )

        new_session_btn.click(
            new_session,
            inputs=[state],
            outputs=[chatbot, state, system_msg, state]
        )

    return demo




def main():
    """Main function to initialize and launch the interface"""
    # Initialize services
    openai.api_key = os.getenv("OPENAI_API_KEY")
    llm_predictor = LLMPredictor(llm=OpenAIChat(
        model_name="gpt-4o",
        temperature=1,
        max_tokens=1024,
        top_p=0.95,
        frequency_penalty=0.4,
        presence_penalty=0.2
    ))
    
    # Configure service context
    service_context = ServiceContext.from_defaults(
        llm_predictor=llm_predictor,
        prompt_helper=PromptHelper(
            max_input_size=4096,
            num_output=256,
            max_chunk_overlap=20
        )
    )
    
    # Create and launch interface
    demo = create_gradio_interface(service_context, api_keys)
    demo.launch(
        server_name="localhost",
        server_port=7860,
        share=False
    )

if __name__ == "__main__":
    main()