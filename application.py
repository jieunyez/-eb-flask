from flask import Flask
from flask import request, json
import pandas as pd
import numpy as np
import math
import nltk
nltk.download('all')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize,RegexpTokenizer
from nltk.stem import WordNetLemmatizer, PorterStemmer

from collections import defaultdict, Counter

# print a nice greeting.
def say_hello(username = "World"):
    return '<p>Hello %s!</p>\n' % username

# some bits of text for the page.
header_text = '''
    <html>\n<head> <title>EB Flask Test</title> </head>\n<body>'''
instructions = '''
    <p><em>Hint</em>: This is a RESTful web service! Append a username
    to the URL (for example: <code>/Thelonious</code>) to say hello to
    someone specific.</p>\n'''
home_link = '<p><a href="/">Back</a></p>\n'
footer_text = '</body>\n</html>'

# EB looks for an 'application' callable by default.
application = Flask(__name__)

file_path = "./search_data/amzProds.csv"
prods_data = pd.read_csv(file_path)


sw = set(stopwords.words('english'))
print(sw)
NUM_TOP_QUESTIONS = 6

reg_tokenizer = RegexpTokenizer("[a-zA-Z]{2,}")
wnl = WordNetLemmatizer() #wnl.lemmatize(t)
ps = PorterStemmer() #ps.stem(t)

def tokenize(doc):
  return [ps.stem(t) for t in reg_tokenizer.tokenize(doc) if not t in sw]

def build_inverted_index(data):

  inverted_index = defaultdict(list)
  for index, dat in data.iterrows():
    dat_vec = tokenize(dat['Product Name'].lower())
    term_freq_in_dat = Counter(dat_vec)
    for term in term_freq_in_dat:
      inverted_index[term].append((index, term_freq_in_dat[term]))            
  return inverted_index
inv_idx = build_inverted_index(prods_data)
index_to_title = {idx:dat['Product Name'] for idx,dat in prods_data.iterrows()}

def compute_idf(inv_idx, n_questions, min_df=15, max_df_ratio=0.90):
    
  idf = {}
  threshold = n_questions*max_df_ratio
  for term in inv_idx:
    if len(inv_idx[term]) > threshold or len(inv_idx[term]) < min_df:
      continue
    idf[term] = math.log2(n_questions / (1 + len(inv_idx[term])))
  return idf
idf = compute_idf(inv_idx, len(prods_data), min_df=5, max_df_ratio=0.7)

def compute_question_norms(index, idf, n_questions):
  norms = np.zeros(n_questions)
  for term in index:
    if term not in idf: 
      continue
    for term_inv_idx in index[term]:
      doc_id = term_inv_idx[0]
      term_freq_doc = term_inv_idx[1]
      norms[doc_id] += (term_freq_doc*idf[term])**2
  return np.sqrt(norms)

inv_idx = {key: val for key, val in inv_idx.items()
           if key in idf} 
question_norms = compute_question_norms(inv_idx, idf, len(prods_data))





# add a rule for the index page.

#@application.route('/')
#def hello():
    #return "Hi"

@application.route('/', methods=['GET'])
def search():
    query = request.args.get('search')

    if not query:
        query = ''
        topQuestionsNoVote = []
    #topQuestionsVote = []
    #topHints = []

    else:
    # Type: [(title, score)], not sorted by score.
    #similarity_score_list = compute_cosine_similarity(query, leetcode_data)
        similarity_score_list = compute_cosine_similarity_tf_idf(query)
        similarity_score_list.sort(key=lambda x: x[1], reverse=True)
        print(similarity_score_list[:5])
    # Type: [(title, url, score, difficulty, description, likes, dislikes)], sorted by score.
        topQuestionsNoVote = [t for t, s in similarity_score_list[:NUM_TOP_QUESTIONS]]

    #sim_score_list_with_vote = sorted([(t, getScoreMultiplier(titleToLike[t], titleToDislike[t])) for t, s in similarity_score_list[:NUM_TOP_QUESTIONS]], key = lambda x: x[1], reverse=True)
    # Type: [(title, url, score, difficulty, description, likes, dislikes)], sorted by score.
    #topQuestionsVote = [(t, titleToURL[t], s, titleToDifficulty[t], titleToDescription[t], titleToLike[t], titleToDislike[t]) for t, s in sim_score_list_with_vote]

    # Type: [(hint, score, summary, url)], sorted by score.
    # topHints = getSortedTopTags(topQuestions)
    #topHints = getSortedTopTagsML(query)
  
    """
    print()
    print([(q, s) for q, _, s, _, _, _, _ in topQuestions]) 
    print()
    print([(t, s) for t, s, _, _ in topHints])
    print()
    """

    objdict= {'ans': topQuestionsNoVote}
    response = application.response_class(
        response=json.dumps(objdict),
        status=200,
        mimetype='application/json'
    )
    #' <br>'.join(topQuestionsNoVote)
    return response

def compute_cosine_similarity_tf_idf(query):    
    index = inv_idx
    res = []
    query_toks = tokenize(query.lower())
    count_q = dict(Counter(query_toks))
    temp_score = defaultdict(lambda:0) # key: doc_id, value: q*d
    for query_term in count_q:
        if query_term not in index:
            continue
        query_term_indexs = index[query_term]
        for term_idx in query_term_indexs:
            doc_id = term_idx[0]
            term_freq = term_idx[1]
            temp_score[doc_id] += count_q[query_term] * idf[query_term] * term_freq * idf[query_term]     
    
    query_norms = 0
    for term in count_q:
        if term not in idf: 
            continue
        query_norms += (count_q[term]*idf[term])**2
    query_norms = math.sqrt(query_norms)
    for doc_id in temp_score:
        temp_score[doc_id] /= question_norms[doc_id]*query_norms
    
    score = [(index_to_title[k], v) for k, v in temp_score.items()]
    return score #sorted(score, key=lambda x:-x[1] )


# run the app.
if __name__ == "__main__":
    # Setting debug to True enables debug output. This line should be
    # removed before deploying a production app.
    #application.debug = True
    application.run()