# FakeOut (Fake News Detector) - SteelHacks 2021

### Inspiration
Information Technology is increasingly utilized to spread false or misleading information. Since it can be difficult for readers to distinguish between real and fake articles at first glance, our Natural Language Processing (NLP) model does the job for them!

### What it does
When the user suspects an online article of being fake, they can pass the article URL to the program. The program uses web scraping to gather the article headline and contents, then passes the text to a BERT-based NLP model which predicts whether the article is real or fake, as well as the percentage likelihood.

### How we built it
The BERT model is based off of [this Kaggle Notebook](https://www.kaggle.com/clmentbisaillon/classifying-fake-news-with-bert) with slight variations. The model is [trained in Google Colab](https://colab.research.google.com/drive/1LVZtft0NidYrTJdpysa9cCNhU3G4Fj6G?usp=sharing), and the trained weights file is downloaded. A local Python script reconstructs the model and loads the weights file so it can make predictions based on the input news article. Finally, the UI is updated with the model outputs.

### Challenges we ran into
Our main challenges include team members' lack of sleep, and the Flask backend not interfacing properly with the model. However, with encouragement and support, we finished the project on time despite the sleep deprivation. In addition, we were able to diagnose and fix the Flask error after some Googling.

### What's next for Fakeout
We are looking at building a mobile version/chrome extension, thus making Fakeout more convenient and easy to use. As well, we want to expand our model to analyze audio and video contexts, thus expanding fake news detection beyond texual media. Lastly, the model could be retrained with a larger and more diverse dataset to increase accuracy.

### Try it out
[colab.research.google.com](https://colab.research.google.com/drive/14GiG6cYyRh9AkCfvj7ceseL2DMTi2B7_?usp=sharing)
