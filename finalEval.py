# import nltk
import string
import os
import PyPDF2

from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from nltk.stem.porter import PorterStemmer

from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from openpyxl import Workbook

# nltk.download('punkt')
# nltk.download('stopwords')

# load stopwords
stop_words = set(stopwords.words('english'))

ps = PorterStemmer()

# paths
answer_path = r'files\ML_Assignment_100pct_correct.pdf'
folder_path = r'files'
model_folder_path = 'offline_model'
result_path = r'result.xlsx'


# pdf extraction
def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            content = page.extract_text()
            if content:
                text += content + "\n"
    return text


# read folder
def folder_reader(folder_path):
    st_names = []
    st_ans = []
    for filename in sorted(os.listdir(folder_path)): 
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(folder_path, filename)
            st_names.append(filename)
            st_ans.append(extract_text_from_pdf(pdf_path))
    return st_names, st_ans


# preprocessing
def preprocessing(text):
    if not isinstance(text, str):
        text = str(text)
    text = text.lower() 
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = " ".join(word for word in text.split() if word not in stop_words)
    text = " ".join(ps.stem(word) for word in text.split())
    return text 


# embedding
def semantic_emb(text_list):
    return model.encode(text_list, convert_to_tensor=True)


# cosine sim
def cos_sim_score(emb_key, emb_ans):
    return util.cos_sim(emb_ans, emb_key).cpu().numpy().flatten().tolist()


# tf idf sim
def tfidf_similarity(text, text_list):
    corpus = [text] + text_list

    vectorizer = TfidfVectorizer(ngram_range=(1, 2))

    vectors = vectorizer.fit_transform(corpus)

    input_vec = vectors[0]
    list_vecs = vectors[1:]

    similarities = cosine_similarity(input_vec, list_vecs).flatten()

    return similarities.tolist()


# sentence level similarity
def sentence_sim_for_each(emb_key, answer):
    sentences = sent_tokenize(answer)

    if len(sentences) == 0:
        return 0

    emb_ans = model.encode(sentences, convert_to_tensor=True)
    sim_matrix = util.cos_sim(emb_ans, emb_key)
    sim_matrix[sim_matrix < 0.6] = 0  
    return sim_matrix.max(dim=1).values.mean().item()


def sentence_level_sim(key, ans):
    key_sentences = sent_tokenize(key)

    emb_key = model.encode(key_sentences, convert_to_tensor=True)

    results = []
    for a in ans:
        results.append(sentence_sim_for_each(emb_key, a))
    return results


# scoring
def give_marks(bert_score, tfidf_score, sentence_score, answers,key):
    marks = []
    for i in range(len(bert_score)):
        score = (0.35 * bert_score[i] +
                 0.15 * tfidf_score[i] +
                 0.5 * sentence_score[i])
        
        lf = length_factor(answers[i], key)
        score = score * lf

        marks.append(min(round(score * 10), 10))

    return marks


# length factor
def length_factor(answer, key):
    ans_len = len(answer.split())
    key_len = len(key.split())

    if key_len == 0:
        return 1

    ratio = ans_len / key_len

    if ratio < 0.5:   # too short 
        return ratio 

    elif 0.5 <= ratio <= 1.5:  # fine
        return 1

    else:        # too long 
        return 1.5 / ratio  

#--------------------------------------------------------------------
#        main execution
#--------------------------------------------------------------------


# extract data
key = extract_text_from_pdf(answer_path)
names, answers = folder_reader(folder_path)

# model
try:
    model = SentenceTransformer(model_folder_path)
except:
    model = SentenceTransformer('all-MiniLM-L6-v2')


# SEMANTIC SIMILARITY
emb_key = model.encode(key, convert_to_tensor=True)
emb_answers = semantic_emb(answers)
semantic_score = cos_sim_score(emb_key, emb_answers)


# SENTENCE SIMILARITY
sentence_score = sentence_level_sim(key, answers)


# preprocessing for tfidf
key_clean = preprocessing(key)
answers_clean = [preprocessing(a) for a in answers]


# TF-IDF
tfidf_score = tfidf_similarity(key_clean, answers_clean)

# scoring
marks = give_marks(semantic_score, tfidf_score, sentence_score,answers,key)


# output
print("Semantic:", semantic_score)
print("TF-IDF:", tfidf_score)
print("Sentence:", sentence_score)
print("Marks:", marks)


# save to execl
wb = Workbook()
ws = wb.active

ws.append(["Name", "Semantic", "TF-IDF", "Sentence", "Final Score"])

for i in range(len(marks)):
    ws.append([names[i], semantic_score[i], tfidf_score[i], sentence_score[i], marks[i]])

wb.save(result_path)