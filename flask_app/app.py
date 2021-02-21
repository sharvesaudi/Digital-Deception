from flask import Flask,render_template,url_for,request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from newspaper import Article
import os


app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        input_url = request.form['input_url']
        news = Article(input_url, language='en')
        news.download()
        news.parse()
        web_text = news.text
        ''' f = open("news.txt", "w")
        f.write(web_text)
        f.close() '''
        pred = test(web_text)
    return render_template('result.html',prediction = pred)



if __name__ == '__main__':
    real = pd.read_csv('./Fake_and_real_news_dataset/True.csv')
    fake = pd.read_csv('./Fake_and_real_news_dataset/Fake.csv')

    # Make both sets the same length
    cutoff = min(len(real), len(fake))
    real = real[:cutoff]
    fake = fake[:cutoff]

    # Add classification label
    real['fake'] = False
    fake['fake'] = True

    # Merge the sets and shuffle order
    df = pd.concat([real, fake])
    df = shuffle(df).reset_index(drop = True)

    # Split into training, validation, and testing
    train, val, test = np.split(df.sample(frac = 1), [int(0.6 * len(df)), int(0.8 * len(df))])

    train = train.reset_index(drop = True)
    val = val.reset_index(drop = True)
    test = test.reset_index(drop = True)

    del real; del fake

    # Import pre-trained BERT model for transfer learning
    # Note: Enable GPU here if available
    device =  torch.device('cpu')
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased")

    model.config.num_labels = 1

    # Freeze pre-trained weights and add 3 new layers
    for param in model.parameters():
        param.requires_grad = False

    model.classifier = nn.Sequential(nn.Linear(768, 256),
                                    nn.ReLU(),
                                    nn.Linear(256, 64),
                                    nn.ReLU(),
                                    nn.Linear(64, 2),
                                    nn.Softmax(dim = 1))

    model = model.to(device)

    criterion = nn.MSELoss().to(device)
    optimizer = optim.SGD(model.classifier.parameters(), lr = 0.01)

    def preprocess_text(text):
        parts = []
        text_len = len(text.split(' '))
        delta = 300
        max_parts = 5
        nb_cuts = int(text_len / delta)
        nb_cuts = min(nb_cuts, max_parts)

        for i in range(nb_cuts + 1):
            text_part = ' '.join(text.split(' ')[i * delta : (i+1) * delta])
            parts.append(tokenizer.encode(text_part, return_tensors = "pt",
                                        max_length = 500).to(device))
        return parts

    print_every = 50
    total_loss = 0
    all_losses = []
    CUDA_LAUNCH_BLOCKING = 1
    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    model.load_state_dict(torch.load("trained_model.pt",map_location=torch.device('cpu')))

    # Test accuracy on the test set
    total = len(test)
    correct = 0
    model.eval()

    def test(text):
        text = preprocess_text(text)
        output = torch.zeros((1, 2)).to(device)
        try:
            for part in text:
                if len(part) > 0:
                    output += model(part.reshape(1, -1))[0]
        except RuntimeError:
            print("GPU out of memory, skipping this entry.")

        output = F.softmax(output[0], dim = -1)
        val, res = output.max(0)

        term = "real" if res.item() == 0 else "fake"
        print("{} at {}%".format(term, val.item() * 100))
        return (term,val.item() * 100)
    app.run(host=os.getenv('IP', '0.0.0.0'),port=int(os.getenv('PORT', 4444)))