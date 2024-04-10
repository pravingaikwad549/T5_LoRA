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

questions = ['What is the name of the customer?', 'What is the phone number of the customer?', 'What is the address of the customer?', 'Sentiment?', 'Summarize?', 'What is the topic of the text?']
print("Questions Loaded Successfully")

app = Flask(__name__)

@app.route('/')
def home():
    print("Home Page")
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    global questions
    res = collections.defaultdict(str)
    # print("type:  #########################",type(request.form['c']))
    context = request.form['c']
    name = request.form['name']
    for i, question in enumerate(questions):
        input_ids = tokenizer(f" question: {question} context: {context}", return_tensors='pt').input_ids
        if i == 3:
            model_senti = PeftModel.from_pretrained(model, "static/Sentiment_adaptors", map_location=torch.device('cpu'))
            output = tokenizer.decode(model_senti.generate(inputs = input_ids)[0], max_length =4, skip_special_tokens=True)
            res[question] = output
        elif i == 4:
            model_summ = PeftModel.from_pretrained(model, "static/Summarization_adaptors", map_location=torch.device('cpu'))
            output = tokenizer.decode(model_summ.generate(inputs = input_ids)[0], num_beam= 1, skip_special_tokens=True) 
            res[question] = output
        else:
            model_QA = PeftModel.from_pretrained(model, "static/QA_adaptors", map_location=torch.device('cpu'))
            output = tokenizer.decode(model_QA.generate(inputs = input_ids)[0], max_new_tokens =4, num_beam= 1, skip_special_tokens=True) 
            res[question] = output
    res = [{'Question': key, 'Answer': value} for key, value in res.items()]
    res = pd.DataFrame(res)
    return render_template('index.html', name = name,  res = res.to_html())

if __name__ == '__main__':
    app.run(debug=True)
