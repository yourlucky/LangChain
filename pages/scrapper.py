import streamlit as st
from urllib.request import urlopen, Request
from urllib.error import HTTPError
from bs4 import BeautifulSoup
import time
import os

import pandas as pd

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.storage import LocalFileStore
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import UnstructuredFileLoader


# Streamlit UI
st.title('Irvine Real Estate Listings')
st.write('This app scrapes real estate listings for the Mission-Viejo area.')

# URL of the site to scrape
url = 'https://www.redfin.com/city/12331/CA/Mission-Viejo/filter/include=forsale+mlsfsbo+construction,status=active,viewport=33.65932:33.55023:-117.55646:-117.7573'

#Fetch HTML content from a given URL, handling HTTP errors.
@st.cache_data(ttl=3600,show_spinner="Loading website...")  
def fetch_html(url):
    req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        response = urlopen(req)
        return response.read()
    except HTTPError as e:
        if e.code == 429:
            st.warning("Too many requests. Waiting for 100 seconds before retrying...")
            time.sleep(100)
            return fetch_html(url)
        else:
            st.error(f"HTTP error occurred: {e}")
            return None

#Scrape traditional real estate data from the HTML soup.
@st.cache_data(ttl=3600)  
def traditional_scrape(html):
    soup = BeautifulSoup(html, "html.parser")
    home_cards = soup.find_all("div", class_="HomeCardContainer")
    data=[]
    
    for card in home_cards:
        address_div = card.find("div", class_="bp-Homecard__Address")
        address = address_div.text.strip() if address_div else "N/A"
        
        price_span = card.find("span", class_="bp-Homecard__Price--value")
        price = price_span.text.strip() if price_span else "N/A"
        
        beds_span = card.find("span", class_="bp-Homecard__Stats--beds")
        beds = beds_span.text.strip() if beds_span else "N/A"
        
        baths_span = card.find("span", class_="bp-Homecard__Stats--baths")
        baths = baths_span.text.strip() if baths_span else "N/A"
        
        sqft_span = card.find("span", class_="bp-Homecard__Stats--sqft")
        sqft = sqft_span.text.strip() if sqft_span else "N/A"
        
        lot_size_span = card.find("span", class_="bp-Homecard__Stats--lotsize")
        lot_size = lot_size_span.text.strip() if lot_size_span else "N/A"
        
        url_a = card.find("a", class_="link-and-anchor")
        url = f"redfin.com{url_a.get('href')}" if url_a else "N/A"

        if all(value == "N/A" for value in [address, price, beds, baths, sqft, lot_size, url]):
            continue
        
        
        data.append({
            "price": price,
            "beds": beds,
            "baths": baths,
            "sqft": sqft,
            "lot_size": lot_size,
            "address": address,
            "url": url
        })
    return data

#@st.cache_data(show_spinner="Embedding HTML...")
def embed_html_directly(html_name):
    # 캐시 디렉토리 설정
    cache_dir = LocalFileStore(f"./.cache/scrapper_embeddings/{html_name}")
    file_content = file.read()
    file_path = f"./.cache/scrapper_files/output.txt"
    with open(file_path, "wb") as f:
        f.write(file_content)

    # 문자열을 적절한 크기의 청크로 나눔
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )

    # HTML 컨텐츠를 직접 분할
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)

    # 임베딩 초기화 및 캐시
    embeddings = OpenAIEmbeddings()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)

    # 벡터 스토어 생성 및 검색기 초기화
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()

    return retriever
# Fetch the HTML content
html = fetch_html(url)
if html:
    # Displaying the entire HTML content (caution: can be very large)
    if st.checkbox('Show full HTML content'):
        st.subheader('Full HTML Content')
        st.text(BeautifulSoup(html, "html.parser").prettify())

    # Extract and display specific parts of the HTML (example)
    st.subheader('Traditional Scrape Results')
    traditional = traditional_scrape(html)
    df = pd.DataFrame(traditional)
    st.dataframe(df)

else:
    st.error("Failed to retrieve the HTML content.")

llm = ChatOpenAI(
    temperature=0.1,
)

answers_prompt = ChatPromptTemplate.from_template(
    """
    You are an AI specialized in formatting real estate data from HTML content.
    Your task is to extract and format the following details into a clean, human-readable form:
    Extract and format real estate data as follows:
    - Price: "price"
    - Beds: "beds"
    - Baths: "baths"
    - Sqft: "sqft"
    - Lot Size: "lot_size"
    - Address: "address"
    - URL: "url"

    Please ensure each property detail is clearly labeled and well-organized.

    Context: {context}
"""
)


st.title('GPT crwaling')
st.write('Write the URL of the website you want to scrape.')


url_gpt = st.text_input(
        "Write down a URL",
        placeholder="https://www.irvinecompanyapartments.com/locations/orange-county/irvine.html",
    )
if url_gpt:
    html = fetch_html(url_gpt)
    soup = BeautifulSoup(html, "html.parser")
    directory = "./.cache/scrapper_files/"
    file_name = "output.txt"
    if not os.path.exists(directory):
        os.makedirs(directory)

    #Displaying the entire HTML content (caution: can be very large)
    if st.checkbox('Show full HTML content',key='url_gpt'):
        st.subheader('Full HTML Content')
        st.text(soup.prettify())
        #print(type(soup.prettify() ))
    file_path = os.path.join(directory, file_name)

    with open(file_path, "w", encoding='utf-8') as file:
        file.write(soup.prettify())

    # retriever = embed_html_directly(soup)
    # context = "some context to use"
    # prompt = answers_prompt.format(context=context)

    # chain = ({"context": retriever}| answers_prompt| llm)
    # st.write(chain.invoke())
    


    #st.subheader('Raw HTML (First 1000 characters)')
    #st.text(str(soup)[:1000])

