from flask import Flask, jsonify, request, redirect
from flasgger import Swagger

from translator import Translator

app = Flask(__name__)
swagger = Swagger(app)

print('\n\n')
print('='*100)
print('Starting Translator Service.....')
translator_obj = Translator()
print('Translator Service Started.')
print('='*100)
print('\n\n')

@app.route('/')
def home():
    redirect('/apidocs', code=302)


@app.route('/translate', methods=['POST'])
def run_translation():
    """Endpoint for Translation of input text to other language.
    ---
    parameters:
      - input_text: translation input
        in: formData
        type: string
      - source_lang: source language
        in: formData
        type: string
      - target_lang: target language
        in: formData
        type: string
    responses:
      200:
        description: Predicted label (Contract-Type) and Probabilities
    """
    if request.method == 'POST':
        inp_txt = request.form['input_text']
        src_lang = request.form['source_lang']
        tgt_lang = request.form['target_lang']
        out_txt = translator_obj.translate(inp_txt, src_lang, tgt_lang)
        return jsonify({'response': out_txt})



if __name__ == '__main__':
    app.run(debug=True)

