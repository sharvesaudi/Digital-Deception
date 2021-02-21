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


app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        input_url = request.form['input_url']
        #web scraping
        web_text = """Price spikes, however, would cause demand to wither and some expensive avocados might be leftover, and stores might try to ration avocados, he added.
            "Exactly what the retail strategy would be in this case, I’m not sure. But we would have vastly fewer avocados," Sumner said.
            Just how fast avocados would disappear, if at all, would depend on whether the Trump administration enacts a full or partial border closure. White House economic adviser Larry Kudlow told CNBC he’s looking for ways to keep some commerce flowing.
            "We are looking at different options, particularly if you can keep those freight lanes, the truck lanes, open," he said this week.  
            Ben Holtz owns Rocky H Ranch, a 70-acre family-run avocado farm in northern San Diego County. He agreed avocados would run out within weeks.
            "Mexico is the big player today. California is not. You shut down the border and California can’t produce to meet the demand," Holtz said. "There will be people without their guacamole."
            While Mexico’s avocado harvest is year-round, California’s is limited to April through July. Growers in the state have picked only about 3 percent of what’s expected to be a much smaller crop of about 175 million pounds this year, Holtz said. A heat wave last summer reduced the crop size.
            California’s avocado harvest has averaged approximately 300 million pounds in recent years, according to data from the California Avocado Commission. By contrast, the U.S. has imported more than 1.5 billion pounds of avocados from Mexico annually. Representatives from the commission did not respond to requests for this article.
            Altogether, the U.S. received 43 percent of its fruit and vegetable imports from Mexico in 2016, according to the U.S. Department of Agriculture.
            Also affecting this year’s avocado supply, a California avocado company in March recalled shipments to six states last month after fears the fruit might be contaminated with a bacterium that can cause health risks.
            Until the early 2000s, California was the nation’s leading supplier of avocados, Holtz said. Mexico gradually overtook the state and now dominates sales in the U.S.
            "It’s a very big possibility," Holtz said of avocado shortages. "Three weeks would dry up the Mexican inventory. California alone consumes more avocados than are grown in our state. Cold storage supply chain is basically three weeks or less of inventory. Most of the time it’s seven days."
            A spokeswoman for the California Restaurant Association said "we haven’t heard concerns from restaurants, it doesn’t mean they aren’t worried." A national grocers association said it will "continue to closely monitor any developments" at the border, but did not have information about the potential impact on avocados.
            """
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
    
    app.run(debug=True)