import json
from tqdm import tqdm
from flask import Flask, request, jsonify

from src.pipeline import Solution

app = Flask(__name__)

obj = Solution()
    

@app.route('/ping')
def app_ping():
    if obj.initial_data_loaded:
        return jsonify(status='ok')
    else:
        return jsonify(status='not ok')

@app.route('/query', methods=["POST"])
def query():
    if not obj.question_loaded:
        return jsonify(status='FAISS is not initialized!')
    else:
        content = json.loads(request.json)
        lang_check = [] # List[bool]
        suggestions = [] # List[Optional[List[Tuple[str, str]]]]
        for doc in tqdm(content['queries']):
            l_check = obj.filter_query(doc)
            lang_check.append(l_check)
            if l_check:
                sugg = obj.get_suggestions(doc)
                # sugg = [('1', doc)]
                suggestions.append(sugg)
            else:
                suggestions.append(None)
        return jsonify(lang_check= lang_check, suggestions=suggestions)
    

@app.route('/update_index', methods=["POST"])
def update_index():
    content = json.loads(request.json) 
    documents = content['documents']
    obj.question_loaded = True
    obj.preprocess_documents(documents)
    obj.create_index()
    index_size = obj.faiss_index.ntotal
    return jsonify(status='ok', index_size=index_size)
        
                       

obj.read_data()