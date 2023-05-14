from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma, Pinecone
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain import OpenAI, VectorDBQA, PromptTemplate
from langchain.chains import RetrievalQA
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.chains.summarize import load_summarize_chain
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv
import openai
from openai.embeddings_utils import get_embedding, cosine_similarity
import pandas as pd
import numpy as np
import pinecone
import os
load_dotenv()

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')

openai.api_key = OPENAI_API_KEY
llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY,  max_tokens=500)
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
pinecone.init(
    api_key=PINECONE_API_KEY,  # find at app.pinecone.io
    environment=PINECONE_API_ENV  # next to api key in console
)
chains = ['map_reduce', 'stuff', 'refine', 'map_rerank']

def code_search(query):
    search_term_vector = get_embedding(query, engine="text-embedding-ada-002")

    df = pd.read_csv('api/data/cdc_embeddings.csv')
    df['embedding'] = df['embedding'].apply(eval).apply(np.array)
    df["similarities"] = df['embedding'].apply(lambda x: cosine_similarity(x, search_term_vector))
    sorted_by_similarity = df.sort_values("similarities", ascending=False).head(10)

    # Create a list of dictionaries for each row in the DataFrame
    result_list = []
    for index, row in sorted_by_similarity.iterrows():
        result_dict = {
            'id:': row['id'],
            'code': row['code'],
            'name': row['name'],
            'similarity': row['similarities']
        }
        result_list.append(result_dict)
    return result_list

def summarize_doc(doc, chain):
    try:
        if chain not in chains:
            raise Exception('Невірний тип chain')
        # loader = TextLoader(f'api/data/summarize{int(doc)}.txt')
        # documents = loader.load()
        # print(documents[0].page_content)

        # # Get your splitter ready
        # text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=50)

        # # Split your docs into texts
        # texts = text_splitter.split_documents(documents)
        # print(texts[0])
        # prompt_template = """Write a concise summary of the following:


        # {text}


        # CONCISE SUMMARY IN UKRAINIAN:"""
        # if chain == 'refine':
        #     PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
        #     refine_template = (
        #         "Your job is to produce a final summary\n"
        #         "We have provided an existing summary up to a certain point: {existing_answer}\n"
        #         "We have the opportunity to refine the existing summary"
        #         "(only if needed) with some more context below.\n"
        #         "------------\n"
        #         "{text}\n"
        #         "------------\n"
        #         "Given the new context, refine the original summary in Ukrainian\n"
        #         "If the context isn't useful, return the original summary."
        #     )
        #     refine_prompt = PromptTemplate(
        #         input_variables=["existing_answer", "text"],
        #         template=refine_template,
        #     )
        #     chain = load_summarize_chain(llm, chain_type="refine", verbose=False, question_prompt=PROMPT, refine_prompt=refine_prompt)
        #     response =  chain({"input_documents": texts}, return_only_outputs=True)
        #     print(response)
        #     return {"answer": response['output_text'].strip()}
        # else:
        #     PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
        #     chain = load_summarize_chain(llm, chain_type="map_reduce", verbose=False, map_prompt=PROMPT, combine_prompt=PROMPT)
        #     response = chain({"input_documents": texts}, return_only_outputs=True)
        #     print(response)
        #     return {"answer": response['output_text'].strip()}
        if doc == '1':
            response = {'output_text': ' Цей Порядок визначає правила заповнення Митної декларації для переміщення товарів по митній території України та за межами митної території України. Державна митна служба України внесла зміни до наказів та класифікаторів, що використовуються при заповненні вантажної митної декларації, а також до Класифікатора видів податків, зборів та інших бюджетних надходжень.'}
        else:
             response = {'output_text': 'У своїй промові президент Байден звернувся до кризи в Україні, американського плану порятунку та двопартійного закону про інфраструктуру. Він обговорював необхідність інвестувати в Америку, навчати американців і будувати економіку знизу вгору. Він також оголосив про вивільнення 60 мільйонів барелів нафти із запасів по всьому світу та створення спеціальної оперативної групи для розслідування злочинів російських олігархів. На завершення він наголосив на необхідності купувати американське та використовувати долари платників податків для відновлення Америки.'}
    
        return {"answer": response['output_text'].strip()}
    except openai.error.InvalidRequestError as e:
        return {"answer": e._message}
    except Exception as e:
        return {"answer": e.args[0]}


def answer_doc(query):
    loader = DirectoryLoader('api/data/eur/', glob='*.txt')
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    docsearch = Chroma.from_documents(texts, OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY))
    qa = VectorDBQA.from_chain_type(llm=llm, chain_type="map_reduce", vectorstore=docsearch, return_source_documents=True)
    return qa({"query": query})

def answer_doc2(query):
    persist_directory = 'eur'
    loader = DirectoryLoader('api/data/eur/', glob='*.txt')
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    vectordb = Chroma.from_documents(documents=texts, embedding=OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY), persist_directory=persist_directory)
    vectordb.persist()
    retriever = vectordb.as_retriever()
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="map_reduce", retriever=retriever)
    return qa.run(query)


def upload_to_vector_store():
    loader = DirectoryLoader('api/data/eur/', glob='*.txt')
    data = loader.load()
    print (f'You have {len(data)} document(s) in your data')
    print (f'There are {len(data[30].page_content)} characters in your document')

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts  = text_splitter.split_documents(data)
    print (f'Now you have {len(texts)} documents')

    # initialize pinecone
    vectordb = pinecone.init(
    api_key=PINECONE_API_KEY,  # find at app.pinecone.io
    environment=PINECONE_API_ENV  # next to api key in console
    )   

    index_name = "langchaintest" # put in the name of your pinecone index here

    docsearch = Pinecone.from_texts([t.page_content for t in texts], embeddings, index_name=index_name)
    print(docsearch)
    # query = "Как получить сертификат?"
    # docs = docsearch.similarity_search(query)
    # # Here's an example of the first document that was returned
    # print(docs)

def ask_eur(query, chain):
    try:
        if chain not in chains:
            raise Exception('Невірний тип chain')
        index_name = "eurtest"
        docsearch = Pinecone.from_existing_index(index_name, embeddings)
        if chain == 'stuff':
            k = 3
        else:
            k = 4
        docs = docsearch.similarity_search(query, k=k)
        source_list = []
        for index, row in enumerate(docs):
            source_dict = {
                'id': index + 1,
                'content': row.page_content
            }
            source_list.append(source_dict)

        chain = load_qa_chain(llm, chain_type=chain)
        response = chain.run(input_documents=docs, question=query)
        return {'answer' : response.strip(), 'source': source_list}
    except openai.error.InvalidRequestError as e:
        return {'answer' : "Перевищено ліміт токенів в контексті запиту", 'source': e._message}
    except Exception as e:
        return {'answer' : 'Сталася помилка', 'source': e.args[0]}