import json
import os
import tiktoken  # اضافه شده برای رفع مشکل توکنایزر

from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    PromptHelper,
    Settings,  # جایگزین ServiceContext
    Document,
    StorageContext,
    load_index_from_storage
)

# استفاده مستقیم از مدل OpenAI مخصوص LlamaIndex برای جلوگیری از تداخل
from llama_index.llms.openai import OpenAI

# اگر نیاز به امبدینگ خاصی دارید (مثلا از OpenAI)، آن را هم اینجا اضافه کنید
# from llama_index.embeddings.openai import OpenAIEmbedding

def setup_global_settings():
    """
    این تابع تنظیمات کلی LlamaIndex را انجام می‌دهد.
    جایگزین تابع قدیمی build_service_context شده است.
    """
    
    # 1. رفع مشکل Tokenizer (خطای tiktoken را حل می‌کند)
    try:
        Settings.tokenizer = tiktoken.get_encoding("cl100k_base").encode
    except Exception as e:
        print(f"Warning: Could not set tokenizer: {e}")

    # 2. تنظیم LLM (جایگزین ChatOpenAI لنچین)
    # ماژول OpenAI در LlamaIndex پارامترهای مشابه دارد
    Settings.llm = OpenAI(
        model="gpt-4o",
        temperature=1.0,
        max_tokens=1024,
        api_key=os.environ.get("GAPGPT_API_KEY"),
        api_base="https://api.gapgpt.app/v1",
        additional_kwargs={
            "top_p": 0.95,
            "frequency_penalty": 0.4,
            "presence_penalty": 0.2
        }
    )

    # 3. تنظیم Prompt Helper (با نام‌های جدید پارامترها)
    Settings.prompt_helper = PromptHelper(
        context_window=4096,        
        num_output=256,             
        chunk_overlap_ratio=0.1     
    )
    
    # 4. تنظیم اندازه Chunk
    Settings.chunk_size = 512
    Settings.chunk_overlap = 20
    
    # اگر نیاز به تنظیم مدل Embedding دارید:
    # Settings.embed_model = OpenAIEmbedding(api_key=..., api_base=...)


def generate_memory_docss(data, language):
    all_user_memories = {}
    for user_name, user_memory in data.items():
        all_user_memories[user_name] = []
        if "history" not in user_memory:
            continue
        for date, content in user_memory["history"].items():
            memory_str = (
                f"日期{date}的对话内容为："
                if language == "cn"
                else f"Conversation on {date}："
            )
            for dialog in content:
                query = dialog["query"]
                response = dialog["response"]
                memory_str += f"\n{user_name}：{query.strip()}"
                memory_str += f"\nAI：{response.strip()}"
            memory_str += "\n"
            if "summary" in user_memory and date in user_memory["summary"]:
                summary = (
                    f'时间{date}的对话总结为：{user_memory["summary"][date]}'
                    if language == "cn"
                    else f'The summary of the conversation on {date} is: {user_memory["summary"][date]}'
                )
                memory_str += summary

            all_user_memories[user_name].append(Document(text=memory_str))
    return all_user_memories


index_set = {}

# تابع قدیمی (اصلاح شده جهت احتیاط)
def build_memory_indexx(all_user_memories, data_args, name=None):
    all_user_memories = generate_memory_docs(
        all_user_memories, data_args.language
    )
    
    # اعمال تنظیمات گلوبال
    setup_global_settings()

    for user_name, memories in all_user_memories.items():
        if name and user_name != name:
            continue
        print(f"build index for user {user_name}")
        
        # حذف آرگومان service_context
        cur_index = VectorStoreIndex.from_documents(memories)
        
        index_set[user_name] = cur_index
        
        # اصلاح مسیر ذخیره‌سازی (متد جدید persist نیاز به دایرکتوری دارد)
        save_dir = f"../memories/memory_index/llamaindex/{user_name}_store"
        os.makedirs(save_dir, exist_ok=True)
        cur_index.storage_context.persist(persist_dir=save_dir)


def generate_memory_docs(data, language):
    all_user_memories = {}

    for user_name, user_memory in data.items():
        all_user_memories[user_name] = {
            "sessions": [],
            "episodic_memory": [],
            "semantic_memory": [],
        }

        if "sessions" in user_memory:
            for session in user_memory["sessions"]:
                date = session["date"]
                content = session["conversation"]

                memory_str = f"Session on {date}:\n"
                for dialog in content:
                    query = dialog["query"]
                    response = dialog["response"]
                    memory_str += f"\n{user_name}: {query.strip()}"
                    memory_str += f"\nAI: {response.strip()}"
                memory_str += "\n"

                if "summary" in user_memory and date in user_memory["summary"]:
                    summary = (
                        f'The summary of the conversation on {date} is: {user_memory["summary"][date]}'
                    )
                    memory_str += summary

                # استفاده از آرگومان text برای Document
                all_user_memories[user_name]["sessions"].append(Document(text=memory_str))


        if "episodic_memory" in user_memory:
            episodic_str = "Recent experiences:\n"
            for event in user_memory["episodic_memory"]:
                episodic_str += f"- {event}\n"
            all_user_memories[user_name]["episodic_memory"].append(Document(text=episodic_str))

        if "semantic_memory" in user_memory:
            semantic_str = "Long-term personality traits:\n"
            if isinstance(user_memory["semantic_memory"], dict):
                for trait, value in user_memory["semantic_memory"].items():
                    semantic_str += f"- {trait}: {value}\n"
            elif isinstance(user_memory["semantic_memory"], list):
                for item in user_memory["semantic_memory"]:
                    semantic_str += f"- {item}\n"
            else:
                print(
                    "WARNING: semantic_memory has an unexpected format:",
                    type(user_memory["semantic_memory"]),
                )

            all_user_memories[user_name]["semantic_memory"].append(Document(text=semantic_str))

    return all_user_memories


# تابع اصلی مورد استفاده
def build_memory_index(all_user_memories, data_args, name=None):
    # 1. تولید داکیومنت‌ها
    all_user_memories = generate_memory_docs(
        all_user_memories, data_args.language
    )
    
    # 2. اعمال تنظیمات جدید (جایگزین ساخت service_context)
    setup_global_settings()

    for user_name, memories in all_user_memories.items():
        if name and user_name != name:
            continue

        print(f"Building indices for user {user_name}...")

        for memory_type, docs in memories.items():
            if not docs:
                continue

            print(f"  → Building {memory_type} index...")
            
            # 3. ساخت ایندکس بدون آرگومان service_context
            # (به صورت خودکار از Settings که بالا ست کردیم استفاده می‌کند)
            cur_index = VectorStoreIndex.from_documents(docs)

            # 4. ذخیره‌سازی با متد جدید persist
            # توجه: در نسخه جدید، persist یک پوشه می‌گیرد نه یک فایل
            index_dir = f"../memories/memory_index/llamaindex/{user_name}/{memory_type}"
            os.makedirs(index_dir, exist_ok=True)
            
            cur_index.storage_context.persist(persist_dir=index_dir)

            index_set[f"{user_name}_{memory_type}"] = cur_index
