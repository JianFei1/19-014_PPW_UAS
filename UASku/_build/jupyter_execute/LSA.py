#!/usr/bin/env python
# coding: utf-8

# # Latent Semantic Analysis (LSA) _ hay

# # Crawling Data Berita

# sebelum melakukan proses crawling data, pastikan anda sudah menginstall library Scrapy dari python. Jika anda belum menginstall Scrapy anda dapat menginstall nya dengan cara ketikkan "pip install Scrapy" pada cmd

# ## Crawling pertama

# pada proses crawling yang pertama ini, kita akan mengambil link yang ada pada halaman kumpulan judul berita. cara untuk melakukan crawling adalah:
# 1. buat file python (.py) misalkan "crawling1.py".
# 2. copy paste code yang ada dibawah ini. (anda dapat memodifikasi kode ini sesuai dengan link berita yang anda inginkan).
# 3. jalankan file "crawling1.py" dengan cara mengetikkan "scrapy runspider crawling1.py -O link.csv" , untuk yang bagian "link.csv" ini merupakan output file yang anda crawling, karena disini saya menggunakan contoh "link.csv" maka hasil outputnya dalam bentuk file csv.

# In[1]:


import scrapy


class QuotesSpider(scrapy.Spider):
    name = "quotes"

    def start_requests(self):

        arrayData = ['https://pta.trunojoyo.ac.id/c_search/byprod/7']
        for i in range(2, 12):
            inArray = 'https://pta.trunojoyo.ac.id/c_search/byprod/7/' + str(i)
            arrayData.append(inArray)
        for url in arrayData:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        for i in range(1,6):
            yield {
                'link': response.css('#content_journal > ul > li:nth-child(' + str(i) + ') > div:nth-child(3) > a::attr(href)').extract()
            }


# ## Crawling kedua

# Untuk proses crawling yang kedua ini, saya mengambil link website berita hasil dari crawling pertama yang sudah di export dalam bentuk csv. untuk membaca file csv ini saya menggunakan library pandas. lalu setelah file dibaca, saya masukkan kedalam array. setelah itu masing masing link akan dilakukan proses crawling.
# Pada proses cawling kedua ini kita akan menuju website beritanya langsung, untuk mendapatkan data judul, label dan isi dari masing-masing berita.
# jalankan file ini dengan cara yang sama seperti yang pertama, akan tetapi sesuaikan nama filenya. cnothnya seperti "scrapy runspider crawling2.py -O isi_berita.csv"

# In[2]:


import scrapy
import pandas as pd



class QuotesSpider(scrapy.Spider):
    name = "quotes"

    def start_requests(self):
        
        dataCSV = pd.read_csv('link_pta.csv')
        indexData = dataCSV.iloc[:, [0]].values
        arrayData = []
        for i in indexData:
            ambil = i[0]
            arrayData.append(ambil)
        for url in arrayData:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        yield {
            'judul': response.css('#content_journal > ul > li > div:nth-child(2) > a::text').extract(),
            'penulis': response.css('#content_journal > ul > li > div:nth-child(2) > div:nth-child(2) > span::text').extract(),
            'dosen_1': response.css('#content_journal > ul > li > div:nth-child(2) > div:nth-child(3) > span::text').extract(),
            'dosen_2': response.css('#content_journal > ul > li > div:nth-child(2) > div:nth-child(4) > span::text').extract(),
            'abstrak_ID': response.css('#content_journal > ul > li > div:nth-child(4) > div:nth-child(2) > p::text').extract(),
            'abstrak_EN': response.css('#content_journal > ul > li > div:nth-child(4) > div:nth-child(4) > p::text').extract(),
            
        }


# # Latent Semantic Analysis (LSA)

# sebelum kita berpindah ke LSA, ada beberapa hal yang perlu dipersiapkan terlebih dahulu.
# beberapa library yang perlu di siapkan yaitu nltk, pandas, numpy dan scikit-learn.
# jika anda menggunakan google colab anda bisa mengetikan syntax dibawah ini untuk melakukan instalasi library yang dibutuhkan.
# 
# !pip install nltk <br>
# !pip install pandas <br>
# !pip install numpy <br>
# !pip install scikit-learn <br>
# 

# ## preprocessing data

# ### import libray
# 
# import library yang dibutuhkan untuk preprocessing data

# In[3]:


# import library
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
import numpy as np


# export file "isi_berita.csv" dalam bentuk data frame pandas.

# In[4]:


#import data frame
dataCSV = pd.read_csv('data_pta.csv')
dataCSV = dataCSV.drop(columns=['penulis', 'dosen_1','dosen_2', 'abstrak_EN'], axis=1)
dataCSV.head()


# ## cek missing value

# In[5]:


dataCSV.isna().sum()


# In[6]:


dataCSV = dataCSV.dropna(axis=0, how='any')


# In[7]:


dataCSV.isna().sum()


# ### Cleansing dan Stopword
# disini kita melakukan cleansing data, yang artinya kita membersihkan data dari simbol, angka dan spasi. <br>
# lalu untuk stopword ini untuk membuang kata yang tidak mempunyai makna seperti:
# 1. "dan"
# 2. "yang" 
# 3. "atau"
# 4. "adalah"

# In[8]:


index_iloc = 0
len_df = len(dataCSV.index)
array_stopwords = []
for kata in range(len_df):
    # indexData itu ambil tiap bagian dari data frame dengan nama dataCSV
    indexData = dataCSV.iloc[index_iloc, [1]].values
    clean_words = [w for w in word_tokenize(indexData[0].lower())
                                    if w.isalpha()
                                    and w not in stopwords.words('indonesian')]
    
    array_stopwords.append(clean_words)
    index_iloc += 1

# membuat kata-kata 1 dokumen di list yang sama
NewArray_stopwords = []
for j in array_stopwords:
    # proses stem per kalimat
    temp = ""
    for i in j:
        # print(i)
        temp = temp +" "+ i

    NewArray_stopwords.append(temp)
print(NewArray_stopwords[0])


# diatas ini adalah contoh isi dari salah satu berita yang sudah dilakukan cleansing dan stopword.

# dibawah ini adalah proses memasukkan data yang sudah dilakukan preprocessing ke dalam data frame yang mempunyai nama "dataSCV"

# In[9]:


dataCSV.head()


# In[10]:


dataCSV = dataCSV.drop('judul', axis=1)
dataCSV = dataCSV.drop('abstrak_ID', axis=1)
dataCSV['isi'] = np.array(NewArray_stopwords)
dataCSV.head()


# ## Term Frequency - Inverse Document Frequency (TF-IDF)

# setelah melakukan pre-processing data, selanjutnya dilakukan proses TF-IDF <br>
# TF-IDF adalah suatu metode algoritma untuk menghitung bobot setiap kata di setiap dokumen dalam korpus. Metode ini juga terkenal efisien, mudah dan memiliki hasil yang akurat. <br>
# Term Frequency (TF) merupakan jumlah kemunculan kata pada setiap dokumen. dirumuskan dengan jumlah frekuensi kata terpilih / jumlah kata <br>
# Inverse Document Matrix (IDF) dirumuskan dengan log((jumlah dokumen / jumlah frekuensi kata terpilih). <br>
# untuk menghasilkan TF-IDF maka hasil dari TF dikalikan dengan IDF, seperti rumus dibawah ini:
# 
# $$
# W_{i, j}=\frac{n_{i, j}}{\sum_{j=1}^{p} n_{j, i}} \log _{2} \frac{D}{d_{j}}
# $$
# 
# Dengan:
# 
# $
# {W_{i, j}}\quad\quad\>: \text { pembobotan tf-idf untuk term ke-j pada dokumen ke-i } \\
# {n_{i, j}}\quad\quad\>\>: \text { jumlah kemunculan term ke-j pada dokumen ke-i }\\
# {p} \quad\quad\quad\>\>: \text { banyaknya term yang terbentuk }\\
# {\sum_{j=1}^{p} n_{j, i}}: \text { jumlah kemunculan seluruh term pada dokumen ke-i }\\
# {d_{j}} \quad\quad\quad: \text { banyaknya dokumen yang mengandung term ke-j }\\
# $
# 
# 

# ### import Library TF-IDF

# import library yang dibutuhkan dalam melakukan pemrosesan TF-IDF dan juga ambil data dari data hasil preprocessing yang sudah dilakukan diatas.

# In[11]:


from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
df = dataCSV


# ### Term Frequency

# ubah data menjadi bentuk list, lalu lakukan proses tf dengan cara memanggil library CountVectorizer dari scikit-learn.

# In[12]:


#mengubah fitur dalam bentuk list
list_isi_berita = []
for i in range(len(df.iloc[:, -1])):
    list_isi_berita.append(df.iloc[i, -1])

# proses term frequency
count_vectorizer = CountVectorizer(min_df=1)
tf = count_vectorizer.fit_transform(list_isi_berita)

#get fitur
fitur = count_vectorizer.get_feature_names_out()

# menampilkan data TF
show_tf = count_vectorizer.fit_transform(list_isi_berita).toarray()
df_tf =pd.DataFrame(data=show_tf,index=list(range(1, len(show_tf[:,1])+1, )),columns=[fitur])
df_tf = df_tf.T

df_tf.head(8)


# ## TF-IDF

# setelah melakukan proses TF, lakukan proses TF-IDF dan kemudian simpan hasilnya dalam bentuk data frame.

# In[13]:


#tfidf dengan tfidf transformer
tfidf_transform = TfidfTransformer(use_idf=True,norm='l2',smooth_idf=True)
tfidf=tfidf_transform.fit_transform(count_vectorizer.fit_transform(list_isi_berita)).toarray()
df_tfidf =pd.DataFrame(data=tfidf,index=list(range(1, len(tfidf[:,1])+1, )),columns=[fitur])
df_tfidf.head(8)


# ## Latent Simantic Analysis (LSA)

# Algoritma LSA (Latent Semantic Analysis) adalah salah satu algoritma yang dapat digunakan untuk menganalisa hubungan antara sebuah frase/kalimat dengan sekumpulan dokumen.
# Dalam pemrosesan LSA ada tahap yang dinamakan Singular Value Decomposition (SVD), SVD adalah salah satu teknik reduksi dimensi yang bermanfaat untuk memperkecil nilai kompleksitas dalam pemrosesan term-document matrix. berikut adalah rumus SVD:
# 
# $$
# A_{m n}=U_{m m} x S_{m n} x V_{n n}^{T}
# $$
# 
# Dengan:
# 
# $
# {A_{m n}}: \text { Matrix Awal } \\
# {U_{m m}}: \text { Matrix ortogonal U }\\
# {S_{m n}}\>: \text { Matrix diagonal S }\\
# {V_{n n}^{T}}\>\>: \text { Transpose matrix ortogonal V }\\
# $

# In[14]:


from sklearn.decomposition import TruncatedSVD


# ### proses LSA dengan library TruncatedSVD dari scikit

# In[15]:


lsa = TruncatedSVD(n_components=8, random_state=36)
lsa_matrix = lsa.fit_transform(tfidf)


# ## proporsi topik pada tiap dokumen

# In[16]:


# menampilkan proporsi tiap topic pada masing-masing dokumen
df_topicDocument =pd.DataFrame(data=lsa_matrix,index=list(range(1, len(lsa_matrix[:,1])+1)))
df_topicDocument.head(6)


# ## proporsi term terhadap topik

# In[17]:


# menampilkan proporsi tiap topic pada masing-masing dokumen
df_termTopic =pd.DataFrame(data=lsa.components_,index=list(range(1, len(lsa.components_[:,1])+1)), columns=[fitur])
df_termTopic.head(100)


# ## Word Cloud

# In[18]:


# most important words for each topic
vocab = count_vectorizer.get_feature_names()

for i, comp in enumerate(lsa.components_):
    vocab_comp = zip(vocab, comp)
    sorted_words = sorted(vocab_comp, key= lambda x:x[1], reverse=True)[:30]
    print("Topic "+str(i)+": ")
    for t in sorted_words:
        print(t[0],end=" ")
    print("\n")


# In[19]:


def draw_word_cloud(index):
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    imp_words_topic=""
    comp=lsa.components_[index]
    vocab_comp = zip(vocab, comp)
    sorted_words = sorted(vocab_comp, key= lambda x:x[1], reverse=True)[:30]
    for word in sorted_words:
        imp_words_topic=imp_words_topic+" "+word[0]

    wordcloud = WordCloud(width=600, height=400).generate(imp_words_topic)
    plt.figure(figsize=(5,5))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


# In[20]:


draw_word_cloud(0)

