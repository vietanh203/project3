# Thư viện xử lý ngôn ngữ tự nhiên
import nltk
# Thư viện xử lý chuỗi
import string
# Thư viện xử lý http requests/response
import requests
# Thư viện xử lý html
from lxml import html
# Thư viện google search
from googlesearch import search
# Thư viện xử lý cấu trúc dữ liệu của đối tương html
from bs4 import BeautifulSoup
# Thư viện scikilearn hỗ trợ tính trọng số Tf-IDF
from sklearn.feature_extraction.text import TfidfVectorizer
# Thư viện scikilearn tính độ tương đồng theo cosine
from sklearn.metrics.pairwise import cosine_similarity

# Thư viện hỗ trợ ngắt, chia thành list câu.
nltk.download('punkt')
# Thư viện hỗ trợ ngắt, chia từ.
nltk.download('wordnet')


def google_query(query, index=0):
    fallback = 'Sorry, I cannot think of a reply for that.'
    result = ''

    try:
        search_result_list = list(search(query, tld="co.in", num=10, stop=3, pause=1))

        page = requests.get(search_result_list[index])

        tree = html.fromstring(page.content)

        soup = BeautifulSoup(page.content, features="lxml")

        article_text = ''
        article = soup.findAll('p')
        for element in article:
            article_text += '\n' + ''.join(element.findAll(text=True))
        article_text = article_text.replace('\n', '')
        first_sentence = article_text.split('.')
        first_sentence = first_sentence[0].split('?')[0]

        chars_without_whitespace = first_sentence.translate(
            {ord(c): None for c in string.whitespace}
        )

        if len(chars_without_whitespace) > 0:
            result = first_sentence
        else:
            result = fallback

        return result
    except:
        if len(result) == 0: result = fallback
        return result


file = open('data/data.txt', 'r', errors='ignore')
rawData = file.read()
rawData = rawData.lower()

sentence_tokens = nltk.sent_tokenize(rawData)
word_tokens = nltk.word_tokenize(rawData)
lemmer = nltk.stem.WordNetLemmatizer()


def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]


remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)


def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))


def getData(user_text):
    bot_response = ''
    sentence_tokens.append(user_text)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words="english")
    tfidf = TfidfVec.fit_transform(sentence_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if (req_tfidf == 0):
        bot_response = google_query(user_text)
        collect_file = open("data/data.txt", "a")
        collect_file.writelines(bot_response)
        collect_file.close()
        return bot_response
    else:
        bot_response = sentence_tokens[idx]
        return bot_response


def getResponse(input):
    user_text = input.lower()
    if (user_text != 'bye'):
        bot_response = getData(user_text)
        sentence_tokens.remove(user_text)
        return str(bot_response)
    else:
        return str('Bye! Thank you!')
