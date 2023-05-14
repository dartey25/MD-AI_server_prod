from flask import Flask, request, jsonify
from models import code_search, ask_eur, summarize_doc

app = Flask(__name__)
app.config['CORS_HEADERS'] = 'Content-Type'


@app.route('/search', methods=['GET'])
def search():
   query = request.args.get('query')
   return jsonify(code_search(query))

@app.route('/eur', methods=['GET'])
def eur():
  query = request.args.get('query')
  chain = request.args.get('chain', 'map_reduce') 
  return jsonify(ask_eur(query=query, chain=chain))

@app.route('/summarize', methods=['GET'])
def summ():
  doc = request.args.get('doc', '1')
  chain = request.args.get('chain', 'map_reduce') 
  return jsonify(summarize_doc(doc=doc, chain=chain))

if __name__ == '__main__':
  app.run(debug=True)