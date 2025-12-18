# -*- coding:utf-8 -*-
import os, shutil
import logging
import sys, openai
import copy
import time, platform

# from typing import TYPE_CHECKING, Any, Callable, Dict, List, Tuple, Type
import signal,json
import gradio as gr
import nltk
import torch
from langchain.llms import AzureOpenAI,OpenAIChat


prompt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
sys.path.append(prompt_path)
from utils.sys_args import data_args,model_args
from utils.app_modules.utils import *
#  
from utils.app_modules.presets import *
from utils.app_modules.overwrites import *
from utils.prompt_utils import *
from utils.memory_utils import enter_name_llamaindex, summarize_memory_event_personality, save_local_memory,extract_session_summary
nltk.data.path = [os.path.join(os.path.dirname(__file__), "nltk_data")] + nltk.data.path

from llama_index import LLMPredictor, GPTSimpleVectorIndex, PromptHelper, ServiceContext
os_name = platform.system()
clear_command = 'cls' if os_name == 'Windows' else 'clear'
stop_stream = False

def signal_handler(signal, frame):
    global stop_stream
    stop_stream = True

VECTOR_SEARCH_TOP_K = 2

api_path ='C:\\Users\\keyva\\MMPL_gpt\\api_key_list.txt'

def read_apis(api_path):
    api_keys = []
    with open(api_path,'r',encoding='utf8') as f:
         lines = f.readlines()
         for line in lines:
             line = line.strip()
             if line:
                 api_keys.append(line)
    return api_keys


memory_dir = os.path.expanduser("C:\\Users\\keyva\\MMPL_gpt\\memories\\update_memory_0512_eng.json")
print(memory_dir)
if not os.path.exists(memory_dir):
    json.dump({},open(memory_dir,"w",encoding="utf-8"))

global memory 
memory = json.load(open(memory_dir,"r",encoding="utf-8"))
language = 'en'
user_keyword = generate_user_keyword()[language]
ai_keyword = generate_ai_keyword()[language]
boot_name = boot_name_dict[language]
boot_actual_name = boot_actual_name_dict[language]
meta_prompt = generate_meta_prompt_dict_chatgpt()[language]
meta_prompt_semantic = generate_meta_prompt_dict_semantic_chatgpt()[language]
new_user_meta_prompt = generate_new_user_meta_prompt_dict_chatgpt()[language]
api_keys = read_apis(api_path)
new_conversation = False  # Start fresh on first entry



deactivated_keys = []
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
)
def create_gradio_interface(user_name, memory, data_args, service_context, api_keys):
    """Creates the Gradio chat interface."""
    with gr.Blocks(title="Psychology AI Assistant") as demo:
        # Store session state
        state = gr.State({
            "history": [],
            "user_name": user_name,
            "memory": memory,
            "data_args": data_args,
            "service_context": service_context,
            "api_keys": api_keys,
            "api_index": 0,
            "semantic_memory_text": None,
            "new_conversation": True
        })
        
        # Header
        gr.Markdown(f"# Psychology AI Assistant\n### Session for {user_name}")
        
        # Chatbot display
        chatbot = gr.Chatbot(label="Conversation")
        
        # User input
        with gr.Row():
            user_input = gr.Textbox(placeholder="Type your message here...", show_label=False)
            submit_btn = gr.Button("Send", variant="primary")
        
        # Additional controls
        with gr.Row():
            clear_btn = gr.Button("Clear Conversation")
            new_session_btn = gr.Button("New Session")
            summary_btn = gr.Button("Summarize Memory")
        
        # System messages
        system_msg = gr.Textbox(label="System Messages", interactive=False)
        
        # Event handlers
        def respond(message, state):
            if not message.strip():
                return state["history"], state["history"], "Empty input.", state
            
            # Call your existing predict function
            history_state, history, msg = predict_new(
                text=message,
                history=state["history"],
                top_p=0.95,
                temperature=1,
                max_length_tokens=1024,
                max_context_length_tokens=200,
                user_name=state["user_name"],
                user_memory=state["memory"],
                user_memory_index=state["memory"].get(state["user_name"], {}).get("sessions", []),
                service_context=state["service_context"],
                api_index=state["api_index"],
                semantic_memory_text=state["semantic_memory_text"],
                query_category=None  # This will be determined in predict_new
            )
            
            # Update state
            new_state = state.copy()
            new_state["history"] = history
            return history_state, new_state, msg, new_state
        
        # Connect UI elements
        submit_btn.click(
            respond,
            inputs=[user_input, state],
            outputs=[chatbot, state, system_msg, state]
        )
        
        # Handle Enter key
        user_input.submit(
            respond,
            inputs=[user_input, state],
            outputs=[chatbot, state, system_msg, state]
        )
        
        # Clear button
        def clear_chat(state):
            new_state = state.copy()
            new_state["history"] = []
            return [], new_state, "Conversation cleared.", new_state
        
        clear_btn.click(
            clear_chat,
            inputs=[state],
            outputs=[chatbot, state, system_msg, state]
        )
        
        # New session button
        def new_session(state):
            # Save current session
            if state["user_name"] in state["memory"]:
                if state["memory"][state["user_name"]]["sessions"]:
                    previous_session = state["memory"][state["user_name"]]["sessions"][-1]
                    session_summary = extract_session_summary(
                        previous_session["conversation"], 
                        previous_session["date"],
                        len(state["memory"][state["user_name"]]["sessions"]) - 1
                    )
                    state["memory"][state["user_name"]]["episodic_memory"].append(session_summary)
            
            # Create new session
            new_session = {
                "session_id": len(state["memory"][state["user_name"]]["sessions"]),
                "date": time.strftime("%Y-%m-%d"),
                "conversation": []
            }
            state["memory"][state["user_name"]]["sessions"].append(new_session)
            
            # Update state
            new_state = state.copy()
            new_state["history"] = []
            new_state["new_conversation"] = True
            return [], new_state, f"🆕 New session started (ID: {new_session['session_id']}).", new_state
        
        new_session_btn.click(
            new_session,
            inputs=[state],
            outputs=[chatbot, state, system_msg, state]
        )
        
        # Summary button
        def summarize_memory(state):
            user_memory = summarize_memory_event_personality(state["data_args"], state["memory"], state["user_name"])
            new_state = state.copy()
            new_state["memory"] = memory  # Update with summarized memory
            return new_state, "Memory summarized successfully.", new_state
        
        summary_btn.click(
            summarize_memory,
            inputs=[state],
            outputs=[state, system_msg, state]
        )
    
    return demo
def chatgpt_chat(prompt,system,history,gpt_config,api_index=0):
        
        retry_times,count = 5,0
        response = None
        while response is None and count<retry_times:
            try:
                request = copy.deepcopy(gpt_config)
                # print(prompt)
                if data_args.language=='en':
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
                message.append({"role":"user","content": f"{prompt}"})
                # print(request)
                # print(message)
                response = openai.ChatCompletion.create(
                    **request, messages=message)
                # print(prompt)
            except Exception as e:
                print(e)
                if 'This key is associated with a deactivated account' in str(e):
                    deactivated_keys.append(api_keys[api_index])
                api_index = api_index+1 if api_index<len(api_keys)-1 else 0
                while api_keys[api_index] in deactivated_keys:
                    api_index = api_index+1 if api_index<len(api_keys)-1 else 0
                openai.api_key = api_keys[api_index]

                count+=1
        if response:
            response = response['choices'][0]['message']['content'] #[response['choices'][i]['text'] for i in range(len(response['choices']))]
        else:
            response = ''
        return response



def classify_query_openai(text, chatgpt_config, api_index=0, retry_times=5):
    """
    Classifies the user's query into one of the memory types: 
    - 'conversation' (specific past discussions)
    - 'episodic_memory' (recent experiences or emotions)
    - 'semantic_memory' (long-term traits, personality, values)
    - 'combined' (broad or ambiguous queries).
    """
   
    response = None
    count = 0
    deactivated_keys = []  # Track deactivated API keys

    system_prompt = """
You are an AI that classifies user queries into one of the following memory types:
- 'conversation' if the query asks about a **specific past discussion** or a **previously mentioned person, place, or event**.
- 'episodic_memory' if the query asks about **recent experiences or emotions**.
- 'semantic_memory' if the query asks about **long-term traits, personality, or values**.
- 'combined' if the query is too broad or ambiguous.

### Instructions:
- **Return ONLY one of these exact labels:** "conversation", "episodic_memory", or "semantic_memory".
- Do NOT include explanations or additional text.
- If a query could belong to multiple categories, return the **most relevant** one.

### Additional Clarifications:
- If the query refers to a **specific person, place, or event** discussed before → classify it as **conversation**.
- If the query asks about **how the user was feeling recently** → classify it as **episodic_memory**.
- If the query asks about **who the user is as a person** → classify it as **semantic_memory**.

### Examples:
User: "What did we talk about last time?"  
AI: conversation  

User: "How have I been feeling lately?"  
AI: episodic_memory  

User: "What kind of person am I?"  
AI: semantic_memory  

User: "Tell me something about me."  
AI: semantic_memory  

User: "Do you know my friend Ali?"  
AI: conversation  

User: "Have we ever talked about my childhood?"  
AI: conversation  

User: "How do I usually react to stress?"  
AI: episodic_memory  

User: "What are my core values?"  
AI: semantic_memory  
"""



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

            # Handle deactivated API key case
            if "This key is associated with a deactivated account" in str(e):
                deactivated_keys.append(api_keys[api_index])

            # Switch API key
            api_index = api_index + 1 if api_index < len(api_keys) - 1 else 0
            while api_keys[api_index] in deactivated_keys:
                api_index = api_index + 1 if api_index < len(api_keys) - 1 else 0
            
            openai.api_key = api_keys[api_index]

            count += 1

    if response:
        category = response["choices"][0]["message"]["content"].strip().lower()
        # Ensure the response is a valid category
        if category not in ["semantic_memory", "episodic_memory", "conversation"]:
            category = "unknown"  # Handle unexpected output
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
  
    chatgpt_config = {"model": "gpt-4o",
        "temperature": temperature,
        "max_tokens": max_length_tokens,
        "top_p": top_p,
        "frequency_penalty": 0.4,
        "presence_penalty": 0.2, 
        'n':1}
    Question_category=classify_query_openai(text,chatgpt_config)
    print("Question_category:",Question_category)
    

    if text == "":
        return history, history, "Empty context."
    system_prompt,related_memo = build_prompt_with_search_memory_llamaindex(
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
        meta_prompt_semantic=meta_prompt_semantic)
    print("system_prompt:++++++++++++++++++++++",system_prompt)
    chatgpt_config = {"model": "gpt-4o",
        "temperature": temperature,
        "max_tokens": max_length_tokens,
        "top_p": top_p,
        "frequency_penalty": 0.4,
        "presence_penalty": 0.2, 
        'n':1
        }
    
    if len(history) > data_args.max_history:
        history = history[data_args.max_history:]
    # print(history)
    response = chatgpt_chat(prompt=text,system=system_prompt,history=history,gpt_config=chatgpt_config,api_index=api_index)
    result = response
    #print('user_memory_index:',user_memory_index)
    #print('prompt:',system_prompt)
    
   
    torch.cuda.empty_cache()
   
    a, b = [[y[0], y[1]] for y in history] + [
                    [text, result]], history + [[text, result]]
    # a, b = [[y[0], convert_to_markdown(y[1])] for y in history] ,history 
    if user_name:
        
        save_local_memory(memory,b,user_name,data_args)
    
    return a, b, "Generating..."

def main(): 
    openai.api_key = os.getenv("OPENAI_API_KEY")
    llm_predictor = LLMPredictor(llm=OpenAIChat(model_name="gpt-4o"))
    max_input_size = 4096
    # set number of output tokens
    num_output = 256
    # set maximum chunk overlap
    max_chunk_overlap = 20
    prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)
    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)
    chatgpt_config = {"model": "gpt-4o",
        "temperature": 0.7,
        "max_tokens": 1024,
        "top_p": 0.9,
        "frequency_penalty": 0.4,
        "presence_penalty": 0.2, 
        'n':1}
    history = []
    global stop_stream
    print('Please Enter Your Name:')
    #user_name = input("\nUser Name：")
    user_name = input("User Name:")
    if user_name in memory.keys():
        if input('Welcome back. Would you like to summarize your memory? If yes, please enter "yes"') == "yes":
            user_memory = summarize_memory_event_personality(data_args, memory, user_name)
    
    
    hello_msg,user_memory,sessions_memory, episodic_memory,semantic_memory  = enter_name_llamaindex(user_name, memory, data_args)
    
    print(hello_msg)
    api_index = 0
    print("Welcome to use SentiSenseBot model，please enter your question to start conversation，enter \"clear\" to clear conversation ，enter \"stop\" to stop program")
    # Initialize a new session
    memo,semantic_memory_text=save_local_memory(memory, history, user_name, data_args, new_conversation=True)
     
     # Create and launch Gradio interface
    demo = create_gradio_interface(
        user_name=user_name,
        memory=memory,
        data_args=data_args,
        service_context=service_context,
        api_keys=api_keys
    )
    
    demo.launch(server_name="0.0.0.0", server_port=7860)


    while True:
        query = input(f"\n{user_name}：")
        
        if  query.strip().lower() == "stop":
            break
        
        if  query.strip().lower() == "clear":
            history = []
            os.system(clear_command)
            print("Welcome to use SiliconFriend model，please enter your question to start conversation，enter \"clear\" to clear conversation ，enter \"stop\" to stop program")
            continue

                 

        # In the main loop where "new session" is handled:
        if query.strip().lower() == "new session":
            # 1. Finalize the previous session (extract episodic memory)
            if memory[user_name]["sessions"]:
                previous_session = memory[user_name]["sessions"][-1]
                session_summary = extract_session_summary(
                    previous_session["conversation"], 
                    previous_session["date"],
                    previous_session["session_id"]  # Pass session ID
                )
                memory[user_name]["episodic_memory"].append(session_summary)

            # 2. Create a new session WITHOUT saving again
            new_session = {
                "session_id": len(memory[user_name]["sessions"]),  # Unique index
                "date": time.strftime("%Y-%m-%d"),
                "conversation": []
            }
            memory[user_name]["sessions"].append(new_session)

            # 3. Reset history
            history = []
            print(f"🆕 New session started (ID: {new_session['session_id']}). Previous session summarized.")
            continue

        if query.strip().lower() == "summary":
            print("Triggering memory summarization...")
            user_memory = summarize_memory_event_personality(data_args, memory, user_name)
            continue
        # Determine which memory to use based on query classification
        query_category = classify_query_openai(query,chatgpt_config)
        #query_category="semantic"

        if query_category == "semantic":
            user_memory_indexx = semantic_memory
        elif query_category == "episodic":
            user_memory_indexx  = episodic_memory
        elif query_category == "conversation":
            user_memory_indexx  = sessions_memory
        else:
            user_memory_indexx  = sessions_memory  # General queries may not need memory
        count = 0
        history_state, history, msg = predict_new(
        text=query,
        history=history,
        top_p=0.95,
        temperature=1,
        max_length_tokens=1024,
        max_context_length_tokens=200,
        user_name=user_name,
        user_memory=user_memory,
        user_memory_index=user_memory_indexx,  # Use the selected memory
        service_context=service_context,
        api_index=api_index,
        semantic_memory_text=semantic_memory_text,
        query_category=query_category

    )
        if stop_stream:
                stop_stream = False
                break
        else:
            count += 1
            if count % 8 == 0:
                #os.system(clear_command)
                print(output_prompt(history_state,user_name,boot_actual_name), flush=True)
                signal.signal(signal.SIGINT, signal_handler)
        #os.system(clear_command)       
        print(output_prompt(history_state,user_name,boot_actual_name), flush=True)
if __name__ == "__main__":
    main()
