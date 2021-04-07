from datasets import load_dataset
dataset = load_dataset("conll2003")
from nltk.tag.util import untag
import sklearn
from sklearn import metrics
'''
print(dataset['train'])
print(dataset['train']['ner_tags'])
print(dataset['train']['tokens'])

tagged_sentences = list()

for i in range(len(dataset['train']['tokens'])):
  merged_list = list(zip(dataset['train']['tokens'][i], dataset['train']['ner_tags'][i]))
  tagged_sentences.append(merged_list[:])
  print(i)


'''
#print(tagged_sentences[0])


def features(sentence, index):
    """ sentence: [w1, w2, ...], index: the index of the word """
    return {
        'word': sentence[index],
        'is_first': index == 0,
        'is_last': index == len(sentence) - 1,
        'is_capitalized': sentence[index][0].upper() == sentence[index][0],
        'is_all_caps': sentence[index].upper() == sentence[index],
        'is_all_lower': sentence[index].lower() == sentence[index],
        'prefix-1': sentence[index][0],
        'prefix-2': sentence[index][:2],
        'prefix-3': sentence[index][:3],
        'suffix-1': sentence[index][-1],
        'suffix-2': sentence[index][-2:],
        'suffix-3': sentence[index][-3:],
        'prev_word': '' if index == 0 else sentence[index - 1],
        'next_word': '' if index == len(sentence) - 1 else sentence[index + 1],
        'has_hyphen': '-' in sentence[index],
        'is_numeric': sentence[index].isdigit(),
        'capitals_inside': sentence[index][1:].lower() != sentence[index][1:]
    }



def transform_to_dataset(tagged_sentences):
    X, y = [], []
 
    for tagged in tagged_sentences:
        X.append([features(untag(tagged), index) for index in range(len(tagged))])
        y.append([str(tag) for _, tag in tagged])
 
    return X, y
 
#X_train, y_train = transform_to_dataset(tagged_sentences)

#print(X_train)
#rint(y_train)

import pickle 
#dbfile = open('X_train', 'ab')
  
# source, destination
#pickle.dump(X_train, dbfile)                     
#dbfile.close()


dbfile = open('X_train', 'rb')     
X_train = pickle.load(dbfile)

dbfile.close()

dbfile = open('y_train', 'rb')     
y_train = pickle.load(dbfile)

dbfile.close()


#dbfile = open('y_train', 'ab')
  
# source, destination
#pickle.dump(y_train, dbfile)                     
#dbfile.close()







from sklearn_crfsuite import CRF
from sklearn_crfsuite import metrics
 
model = CRF()
model.fit(X_train, y_train)

dbfile = open('model', 'ab')
  

pickle.dump(model, dbfile)                     
dbfile.close()

sentence = ['I', 'am', 'Bob','!']
 
def pos_tag(sentence):
    sentence_features = [features(sentence, index) for index in range(len(sentence))]
    return list(zip(sentence, model.predict([sentence_features])[0]))
 
print(pos_tag(sentence))  # [('I', 'PRP'), ('am', 'VBP'), ('Bob', 'NNP'), ('!', '.')]


labels = dataset['train'].features['ner_tags'].feature
num_labels = dataset['train'].features['ner_tags'].feature.num_classes
label2id = {str(id): val for id, val in enumerate(labels.names)}
id2label = {v: k for k, v in label2id.items()}


tagged_sentences = list()

for i in range(len(dataset['test']['tokens'])):
  merged_list = list(zip(dataset['test']['tokens'][i], dataset['test']['ner_tags'][i]))
  tagged_sentences.append(merged_list[:])
  print(i)



X_test, y_test = transform_to_dataset(tagged_sentences)

y_pred = model.predict(X_test)

print(metrics.flat_classification_report(y_test, y_pred))
