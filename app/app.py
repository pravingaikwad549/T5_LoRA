from flask import Flask, render_template, request
import torch
from transformers import (T5ForConditionalGeneration, T5Tokenizer)
from peft import PeftModel
import collections
import pandas as pd
print("Libraries Imported Successfully")
pd.set_option('display.max_colwidth', None)
model_id = 't5-large'
tokenizer = T5Tokenizer.from_pretrained(model_id)
model = T5ForConditionalGeneration.from_pretrained(model_id)

questions = ['What is the name of the customer?', 'What is the phone number of the customer?', 'What is the address of the customer?', 'Sentiment?', 'Summarize:']
print("Questions Loaded Successfully")

app = Flask(__name__)

@app.route('/')
def home():
    print("Home Page")
    return render_template('index.html')

@app.route('/model_details')
def model_details():
    return render_template('model_details.html')

@app.route('/predict', methods=['POST'])
def predict():
    global questions
    res = collections.defaultdict(str)
    context = request.form['c']
    for i, question in enumerate(questions):
        input_ids = tokenizer(f" question: {question} context: {context}", return_tensors='pt').input_ids
        if i == 3:
            model_senti = PeftModel.from_pretrained(model, "static/Sentiment_adaptors", map_location=torch.device('cpu'))
            model_senti.eval()
            output = tokenizer.decode(model_senti.generate(inputs = input_ids)[0], max_length =4, skip_special_tokens=True)
            res[question] = output
        elif i == 4:
            model_summ = PeftModel.from_pretrained(model, "static/Summarization_adaptors", map_location=torch.device('cpu'))
            model_summ.eval()
            output = tokenizer.decode(model_summ.generate(inputs = input_ids)[0], num_beam= 1, skip_special_tokens=True) 
            res[question] = output
        else:
            model_QA = PeftModel.from_pretrained(model, "static/QA_adaptors", map_location=torch.device('cpu'))
            model_QA.eval()
            output = tokenizer.decode(model_QA.generate(inputs = input_ids)[0], max_new_tokens =4, num_beam= 1, skip_special_tokens=True) 
            res[question] = output
    res = [{'Question': key, 'Answer': value} for key, value in res.items()]
    res = pd.DataFrame(res)
    return render_template('index.html', res = res.to_html(), text = context)
@app.route('/custom_question', methods=['GET', 'POST'])
def custom():
    if request.method == 'POST':
        question = request.form.get('q', '')
        context = request.form.get('c', '') 
        model_QA = PeftModel.from_pretrained(model, "static/QA_adaptors", map_location=torch.device('cpu'))
        model_QA.eval()
        input_ids = tokenizer(f" question: {question} context: {context}", return_tensors='pt').input_ids
        output = tokenizer.decode(model_QA.generate(inputs=input_ids)[0], max_new_tokens=4, num_beam=1, skip_special_tokens=True)
        return render_template('custom.html', output=output, text = context, question = question)
    return render_template('custom.html')


if __name__ == '__main__':
    app.run(debug=True, port = 8000)
