## Problem Statement 1:
### Natural Language Processing for Named Entity Recognition (NLP for NER):
Idetitifying and recognising named entity from raw sentence or text is a complex NLP task in machine learning and text engineering. It requires cleaning, data pre-processing, feature engineering, CNN and Kmeans for structing unstructured raw data and many other complex task. 

Identify and predict entities like ORG[Organisation names], LOC[Locations], PER[Person name] and IFSC code[Optional] from given raw data. Most of the companies are working on similar task for text engineering and NER for such complex use cases. This will gives you hands-on experience on NLP and text engineering:

Techinolgies:

1. Can use [spaCy](https://spacy.io) or [BERT](https://github.com/google-research/bert).
2. [spaCy](https://spacy.io) is recommended as it has good accuracy and proper documentation.
3. Python.
4. Google colab.

Process:

1. Research on this topic properly.
2. Use WIKIPEDIA raw data for training or from [here](https://www.clips.uantwerpen.be/conll2003/ner/) or from [this repo](https://github.com/kyzhouhzau/BERT-NER/tree/master/data)
3. Build and train your model.
4. Test and predict yor result.
5. Submit your file for verification on mail.

Submission:

Submit CSV file on your data with prediction in following column format:
_SENTENCE|ENTITYNAME|ENTITYTYPE|CONFIDENCE_

Example:

SENTENCE                            |ENTITYNAME       |ENTITYTYPE|CONFIDENCE
FONTERRA LIMITED is from Australia  |FONTERRA LIMITED |ORG       |0.93
FONTERRA LIMITED is from Australia  |AUSTRALIA        |LOC       |0.95


References:
1. [spaCy](https://spacy.io)
2. [spaCy Learning](https://spacy.io/usage/linguistic-features)
3. [BERT](https://github.com/google-research/bert)
4. [BERT example repo with TF](https://github.com/kyzhouhzau/BERT-NER)
