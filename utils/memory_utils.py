import os, shutil, datetime, time, json
import sys
import time
import gradio as gr
from pprint import pprint
#from llama_index import GPTSimpleVectorIndex
#from llama_index.core import VectorStoreIndex

from llama_index.core import StorageContext, load_index_from_storage, VectorStoreIndex


bank_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../memory_bank')
sys.path.append(bank_path)
from build_memory_index import build_memory_index
memory_bank_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../memory_bank')
sys.path.append(memory_bank_path)
from summarize_memory import summarize_memory,extract_session_summary,extract_semantic_memory

def enter_name(name, memory,local_memory_qa,data_args,update_memory_index=True):
    cur_date = datetime.date.today().strftime("%Y-%m-%d")
    user_memory_index = None
    if isinstance(data_args,gr.State):
        data_args = data_args.value
    if isinstance(memory,gr.State):
        memory = memory.value
    if isinstance(local_memory_qa,gr.State):
        local_memory_qa=local_memory_qa.value
    memory_dir = os.path.expanduser("C:\\Users\\keyva\\MMPL_gpt\\memories\\update_memory_0512_eng.json")
    
    #memory_dir = os.path.join(data_args.memory_basic_dir,data_args.memory_file)
    if name in memory.keys():
        user_memory = memory[name]
        memory_index_path = os.path.join(data_args.memory_basic_dir,f'memory_index/{name}_index')
        os.makedirs(os.path.dirname(memory_index_path), exist_ok=True)
        if (not os.path.exists(memory_index_path)) or update_memory_index:
            print(f'Initializing memory index {memory_index_path}...')
        
            if os.path.exists(memory_index_path):
                shutil.rmtree(memory_index_path)
            memory_index_path, _ = local_memory_qa.init_memory_vector_store(filepath=memory_dir,vs_path=memory_index_path,user_name=name,cur_date=cur_date)                      
        
        user_memory_index = local_memory_qa.load_memory_index(memory_index_path) if memory_index_path else None
        msg = f"Welcome back,{name}！" if data_args.language=='cn' else f"Wellcome Back, {name}！"
        return msg,user_memory,memory, name,user_memory_index
    else:
        memory[name] = {}
        memory[name].update({"name":name}) 
        msg = f"Welcome New Users{name}I will remember your name and call you by your name next time we meet!" if data_args.language == 'cn' else f'Welcome, new user {name}! I will remember your name, so next time we meet, I\'ll be able to call you by your name!'
        return msg,memory[name],memory,name,user_memory_index
#old function
def enter_name_llamaindexx(name, memory, data_args, update_memory_index=True):
    user_memory_index = None
    if name in memory.keys():
        user_memory = memory[name]
        #memory_index_path ="C:\\Users\\keyva\\MMPL_gpt\\memories\\memory_index"

        memory_index_path = os.path.join(data_args.memory_basic_dir,f'../memories/memory_index/llamaindex/{name}_index.json')
        
        if not os.path.exists(memory_index_path) or update_memory_index:
            print(f'Initializing memory index {memory_index_path}...')
            build_memory_index(memory,data_args,name=name)

        if os.path.exists(memory_index_path):
            user_memory_index = VectorStoreIndex.load_from_disk(memory_index_path)
            print(f'Successfully load memory index for user {name}!')
            
            
        return f"Wellcome Back, {name}！",user_memory,user_memory_index
    else:
        memory[name] = {}
        memory[name].update({"name":name}) 
        return f"Welcome new user{name}！I will remember your name and call you by your name in the next conversation",memory[name],user_memory_index

#old function
def enter_name_llamaindexx(name, memory, data_args, update_memory_index=True):
    """
    Loads user memory and separate indices for each memory type (conversation, episodic, semantic).
    """
    user_memory_indices = {"conversation": None, "episodic_memory": None, "semantic_memory": None}

    if name in memory.keys():
        user_memory = memory[name]

        base_path = os.path.join(data_args.memory_basic_dir, f'../memories/memory_index/llamaindex/{name}')
        
        # If indices do not exist or need updating, rebuild them
        if update_memory_index or not all(os.path.exists(f"{base_path}/{mem_type}_index.json") for mem_type in user_memory_indices):
            print(f'Initializing memory indices for {name}...')
            build_memory_index(memory, data_args, name=name)

        # Load each memory type separately
        for mem_type in user_memory_indices:
            index_path = f"{base_path}/{mem_type}_index.json"
            if os.path.exists(index_path):
                user_memory_indices[mem_type] = VectorStoreIndex.load_from_disk(index_path)
                print(f'  → Loaded {mem_type} memory index for {name}.')

        return f"Welcome back, {name}!", user_memory, user_memory_indices

    else:
        memory[name] = {"name": name}  # Initialize new user memory
        return f"Welcome, new user {name}! I'll remember your name for next time.", memory[name], user_memory_indices

def enter_name_llamaindex(name, memory, data_args, update_memory_index=True):
    """
    Loads user memory and separate indices for each memory type (sessions, episodic, semantic).
    Compatible with LlamaIndex v0.10+
    """
    sessions_memory = None
    episodic_memory = None
    semantic_memory = None

    if name in memory.keys():
        user_memory = memory[name]

        # مسیر پایه را تنظیم می‌کنیم
        base_path = os.path.join(data_args.memory_basic_dir, f'../memories/memory_index/llamaindex/{name}')
        
        # نام‌گذاری مسیرها (فرض بر این است که در مرحله ساخت، پوشه‌هایی با این نام‌ها ایجاد شده‌اند)
        # نکته: اگر در مرحله ساخت پسوند .json را حذف کرده‌اید، اینجا هم باید حذف کنید.
        # اما طبق کد شما، فرض می‌کنیم نام فولدرها به .json ختم می‌شود.
        sessions_path = f"{base_path}/sessions_index.json"
        episodic_path = f"{base_path}/episodic_memory_index.json"
        semantic_path = f"{base_path}/semantic_memory_index.json"

        # بررسی وجود دایرکتوری‌ها برای تصمیم‌گیری جهت ساخت مجدد
        indices_exist = (
            os.path.exists(sessions_path) and 
            os.path.exists(episodic_path) and 
            os.path.exists(semantic_path)
        )

        # If indices do not exist or need updating, rebuild them
        if update_memory_index or not indices_exist:
            print(f'Initializing memory indices for {name}...')
            # فرض بر این است که تابع build_memory_index ایمپورت شده و در دسترس است
            build_memory_index(memory, data_args, name=name)

        # --- بخش اصلاح شده: بارگذاری با روش جدید LlamaIndex ---
        
        # 1. بارگذاری حافظه جلسات
        if os.path.exists(sessions_path):
            try:
                storage_context = StorageContext.from_defaults(persist_dir=sessions_path)
                sessions_memory = load_index_from_storage(storage_context)
                print(f'  → Loaded sessions memory index for {name}.')
            except Exception as e:
                print(f"Warning: Could not load sessions memory: {e}")
        
        # 2. بارگذاری حافظه اپیزودیک
        if os.path.exists(episodic_path):
            try:
                storage_context = StorageContext.from_defaults(persist_dir=episodic_path)
                episodic_memory = load_index_from_storage(storage_context)
                print(f'  → Loaded episodic memory index for {name}.')
            except Exception as e:
                print(f"Warning: Could not load episodic memory: {e}")
        
        # 3. بارگذاری حافظه معنایی
        if os.path.exists(semantic_path):
            try:
                storage_context = StorageContext.from_defaults(persist_dir=semantic_path)
                semantic_memory = load_index_from_storage(storage_context)
                print(f'  → Loaded semantic memory index for {name}.')
            except Exception as e:
                print(f"Warning: Could not load semantic memory: {e}")

        return f"Welcome back, {name}!", user_memory, sessions_memory, episodic_memory, semantic_memory

    else:
        # کاربر جدید
        memory[name] = {"name": name}  # Initialize new user memory
        return f"Welcome, new user {name}! I'll remember your name for next time.", memory[name], None, None, None


def summarize_memory_event_personality(data_args, memory, user_name):
    if isinstance(data_args,gr.State):
        data_args = data_args.value
    if isinstance(memory,gr.State):
        memory = memory.value
    memory_dir = os.path.expanduser("C:\\Users\\keyva\\MMPL_gpt\\memories\\update_memory_0512_eng.json")
    memory = summarize_memory(memory_dir,user_name,language=data_args.language)
    user_memory = memory[user_name] if user_name in memory.keys() else {}
    return user_memory #, user_memory_index 
# semantic_memory.py
import json
import time
from typing import Dict, Any


import os
import json
import time
import gradio as gr

def save_local_memory(memory, history, user_name, data_args, new_conversation=False):
    """
    Saves user-model conversations into memory and adds episodic memory for each session.
    Handles both list-of-lists (old Gradio) and list-of-dicts (new Gradio/OpenAI) formats.
    """
    if isinstance(data_args, gr.State):
        data_args = data_args.value
    if isinstance(memory, gr.State):
        memory = memory.value

    # مسیر فایل را متناسب با سیستم خودتان چک کنید
    memory_dir = os.path.expanduser("C:\\Users\\keyva\\MMPL_gpt\\memories\\update_memory_0512_eng.json")

    # 1. Initialize user memory with ALL required structures FIRST
    if user_name not in memory:
        memory[user_name] = {
            "sessions": [],
            "episodic_memory": [],
            "semantic_memory": {}
        }
    
    # 2. Ensure all sub-structures exist and are correct type
    memory[user_name].setdefault("sessions", [])
    memory[user_name].setdefault("episodic_memory", [])
    memory[user_name].setdefault("semantic_memory", {})

    # 3. Now safely check types
    if not isinstance(memory[user_name]["semantic_memory"], dict):
        memory[user_name]["semantic_memory"] = {}

    # Create new session or update existing one
    if new_conversation or not memory[user_name]["sessions"]:
        if new_conversation and memory[user_name]["sessions"]:
            previous_session = memory[user_name]["sessions"][-1]
            # تابع extract_session_summary باید در فایل utils موجود باشد
            # session_summary = extract_session_summary(...)
            # بخش‌های کامنت شده کد اصلی شما حفظ شدند

        # Create new session
        session = {
            "session_id": len(memory[user_name]["sessions"]),
            "date": time.strftime("%Y-%m-%d", time.localtime()),
            "conversation": []
        }
        memory[user_name]["sessions"].append(session)

    current_session = memory[user_name]["sessions"][-1]
    
    # --- بخش اصلاح شده برای رفع خطای KeyError: 0 ---
    if not new_conversation and history:
        last_item = history[-1]
        
        # حالت ۱: فرمت جدید (دیکشنری) - List of Dicts
        if isinstance(last_item, dict):
            # در این فرمت، تاریخچه خطی است. آخرین آیتم جواب ربات و یکی مانده به آخر سوال کاربر است
            if len(history) >= 2:
                user_query = history[-2].get('content', '')
                bot_response = history[-1].get('content', '')
                
                current_session["conversation"].append({
                    'query': user_query, 
                    'response': bot_response
                })
                
        # حالت ۲: فرمت قدیمی (لیست یا تاپل) - List of Lists
        elif isinstance(last_item, (list, tuple)):
            current_session["conversation"].append({
                'query': last_item[0], 
                'response': last_item[1]
            })
    # -----------------------------------------------

    # توجه: مطمئن شوید extract_semantic_memory ایمپورت شده باشد
    # memory[user_name]["semantic_memory"] = extract_semantic_memory(
    #    memory[user_name]["semantic_memory"], 
    #    current_session["conversation"]
    # )
    
    semantic_memory_text = memory[user_name]["semantic_memory"]
    
    # ذخیره در فایل
    # اطمینان حاصل کنید که پوشه memories وجود دارد
    os.makedirs(os.path.dirname(memory_dir), exist_ok=True)
    
    with open(memory_dir, "w", encoding="utf-8") as f:
        json.dump(memory, f, ensure_ascii=False, indent=4)

    return memory, semantic_memory_text
