import os
import ssl
import nltk
import importlib

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader, ServiceContext, Document, StorageContext, load_index_from_storage

def load_knowledge(knowledge_dir):
    documents = SimpleDirectoryReader(knowledge_dir).load_data()
    return documents

def create_index(documents, service_context):
    print('Creating new index')
    index = GPTVectorStoreIndex.from_documents(documents, service_context=service_context)
    save_index(index)
    return index

def save_index(index):
    # Save index to file within the knowledge_dir
    persist_dir = os.path.join(knowledge_dir, 'index.json')
    index.storage_context.persist(persist_dir=persist_dir)

def load_index():
    try:
        # Rebuild storage context
        storage_context = StorageContext.from_defaults(persist_dir=os.path.join(knowledge_dir, 'index.json'))
        # Load index
        index = load_index_from_storage(storage_context)
    except FileNotFoundError:
        index = create_index()
    return index

def query_index(index):
    query_engine = index.as_query_engine()
    while True:
        prompt = input("Type prompt...")
        response = query_engine.query(prompt)
        print(f"\n\n{response}\n\n")

knowledge_dir = ''

def main():
    global knowledge_dir
    party_names = os.listdir('knowledge')
    if not party_names:
        print("No parties available.")
        return

    print("Available parties:")
    for i, party_name in enumerate(party_names, start=1):
        print(f"{i}. {party_name}")

    party_choice = input("Choose a party by entering its number: ")
    party_choice = int(party_choice) - 1

    if party_choice < 0 or party_choice >= len(party_names):
        print("Invalid party choice.")
        return

    party_name = party_names[party_choice]
    knowledge_dir = f'knowledge/{party_name}'
    documents = load_knowledge(knowledge_dir)
    service_context = ServiceContext.from_defaults(chunk_size_limit=3000)

    refresh_index = input(f"Do you want to refresh the {party_name} index? (y/n) [n]: ")
    refresh_index = refresh_index.lower() == 'y'

    if refresh_index:
        index = create_index(documents, service_context)
    else:
        index = load_index()

    query_index(index)

if __name__ == '__main__':
    main()
