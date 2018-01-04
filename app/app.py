## "set FLASK_APP=app.py" in windows
## "export FLASK_APP=app.py" in linux/Max OS
from flask import Flask
from flask import request
from flask import render_template
from flask import abort, redirect, url_for
from wtforms import Form,  StringField, validators
import metapy
import numpy as np
import pandas as pd
app = Flask(__name__)

class RegistrationForm(Form):
    username = StringField('Username')

data = pd.read_csv("people.csv")        
data["name"] = data["name"].str.lower()
model = metapy.embeddings.load_embeddings('./people_emd.toml')
def embed(name):
    names = name.split(' ')
    A = np.zeros((50,))
    for i in names:
        if model.at(i)[0] != 91536:
            A += model.at(i)[1]
    if sum(A) == 0:
        return None, None
    else:
        return name, [model.term(i[0]) for i in model.top_k(A)]

topi = pd.read_csv("topic.csv")    
topi["Name"] = topi["Name"].str.lower()
def topic(name):
    topics = []
    if name in np.array(topi["Name"]):
        if topi[topi["Name"] == name]['Politician'].iloc[0] == 1:
            topics.append('Politician')    
        if topi[topi["Name"] == name]['Scientist'].iloc[0] == 1:
            topics.append('Scientist')
        if topi[topi["Name"] == name]['Athlete'].iloc[0] == 1:
            topics.append('Athlete')
        if topi[topi["Name"] == name]['Artist'].iloc[0] == 1:
            topics.append('Artist')
    return ', '.join(topics)

Topic_model = metapy.topics.TopicModel('lda-cvb4')
fidx = metapy.index.make_forward_index('people-config.toml')
def word_in_topics(topics_id):
    scorer = metapy.topics.BLTermScorer(Topic_model)
    return [(fidx.term_text(pr[0]), pr[1]) for pr in Topic_model.top_k(tid=topics_id, scorer=scorer)]

@app.route('/', methods=['GET', 'POST'])
def search():
    form = RegistrationForm(request.form)
    if request.method == 'POST' and form.validate():
        user = form.username.data
        return redirect(url_for('people', name = user.lower()))
    return render_template('search.html', form=form)


@app.route('/people/<name>')
def people(name=None):
    if name in np.array(data["name"]):
        bios = data[data["name"] == name]['text'].iloc[0]
        name, embedding = embed(name)
        topics = topic(name)
        return render_template('people.html', name=name, topics = topics, bios = bios, embedding = embedding)
    else:
        word, embedding = embed(name)
        return render_template('people.html', name=None, word=word, embedding = embedding)

@app.route('/topic/<topics>')
def distribution(topics=None):
    if topics == "Politician":
	    word_distribution = word_in_topics(0)
    if topics == "Scientist":
	    word_distribution = word_in_topics(1)
    if topics == "Athlete":
	    word_distribution = word_in_topics(2)
    if topics == "Artist":
	    word_distribution = word_in_topics(3)
    return render_template('topic.html', topics = topics, word_distribution = word_distribution)

    
