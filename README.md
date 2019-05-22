# Persistent Systems internship program - 2019
-----

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

## Problem Statement 2:
### Object Detection using Tensorflow:
Object detection using Machine Learning and Deep learning is tredind topic now days. Using Tensorflow, YOLO, scikit learn we can easily detect any object from given image, video or live stream.

Train a model to detect and classify object like 'Helmet' from image, video and live stream with its confidence value. Detecting Helmet will be useful use-case for traffic police, roas safety and at other manufacturing companies where helmets are compulsary for workers.
Accurately automating this process using AI/ML will reduce human efforts, human accidents/casualties.

Techinolgies:
1. Tensorflow Object detection.
2. OpenCV.
3. Python.
4. Google colab.

Process:
1. Research this topic properly.
2. Collect required images from google, cellphone or from kaggle.
3. Build, train and test your model at colab with GPU.
4. Shoot and upload real time demo video at YouTube.

Submission:
1. Your youtube video link for demo video.

References:
1. [TensorFlow](https://www.tensorflow.org/)
2. [Card detection Video](https://www.youtube.com/watch?v=Rgpfk6eYxJA)
3. [Card detection repo](https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10)
4. [Tensorflow for Poets - Model generation](https://codelabs.developers.google.com/codelabs/tensorflow-for-poets/index.html#0)
5. [Tensorflow for Poets - TFlite convert and prediction on Android devices](https://codelabs.developers.google.com/codelabs/tensorflow-for-poets-2-tflite/#0)
6. [My youtube Demo for Cotton-aphid prediction](https://www.youtube.com/watch?v=owTYf5SyRhE) 

_ML hints for your ideas and motivation:_

## A. Object detection:
1. [Creating TFRecords - TensorFlow Object Detection API Tutorial 4-6](https://www.youtube.com/watch?v=kq2Gjv_pPe8)
2. [How To Train an Object Detection Classifier Using TensorFlow 1.5 (GPU) on Windows 10](https://www.youtube.com/watch?v=Rgpfk6eYxJA)


[_These are only to get motivated and come up with your own related ideas._]

3. YOLO [_Search online about it's v3 version_]

    [Demo video](https://www.youtube.com/watch?v=BNHJRRUKMa4)
    
    [Example video by Siraj](https://www.youtube.com/watch?v=4eIBisqx9_g)
4. Deep learning Image Classification

    [Example Video](https://www.youtube.com/watch?v=cAICT4Al5Ow)
    
    [Example code](https://becominghuman.ai/building-an-image-classifier-using-deep-learning-in-python-totally-from-a-beginners-perspective-be8dbaf22dd8)
    
5. ML or OpenCV based traffic related operation

    [Traffic counting 1](https://www.youtube.com/watch?v=z1Cvn3_4yGo)
    
    [Traffic counting 2](https://www.youtube.com/watch?v=O0aZygGcGZE)
    
    [ML OpenCv lane and vehicle detection](https://www.youtube.com/watch?v=pQuUW3Jp8ic)
    
6. ML Natural Language Processing using spaCy

    [Website](https://spacy.io/usage/linguistic-features)
7. OpenCv gesture recognition
 
    [Example](https://www.youtube.com/watch?v=v-XcmsYlzjA)

8. [Playing Card detection using YOLO](https://www.youtube.com/watch?v=pnntrewH0xg)

## B. Natural Language Processing:
1. [spaCy](https://spacy.io)
2. [BERT](https://towardsdatascience.com/bert-in-keras-with-tensorflow-hub-76bcbc9417b)
3. [BERT Illustrated](https://jalammar.github.io/illustrated-bert/)

----

## _Enough motivation, lets learn to implement it._

Pre-requisite:
1. Python
2. Handling python jupyter notebook.
3. Basic ML.
4. Basic OpenCV.

----

Some reference links for learning;

1. [Machine Learning](https://www.kaggle.com/learn/machine-learning)
2. [Python](https://www.kaggle.com/learn/python)

Some youtube channels to follow:
1. Siraj Raval - ML and AI

    [Channel](https://www.youtube.com/channel/UCWN3xxRkmTPmbKwht9FuE5A)    
    [Exampple Video](https://www.youtube.com/watch?v=2FOXR16mLow&list=PL2-dafEMk2A4ut2pyv0fSIXqOzXtBGkLj)

2. Sebastiaan Mathôt - Python

    [Python Lectures](https://www.youtube.com/watch?v=rytP_vIjzeE&list=PLR-r0edywujd8D-R2Kue1C_wYEK_4Ii71&index=16)
    
Online ML practice environment:
1. [Google colabs](https://colab.research.google.com/)
2. [Floydhub](https://floydhub.com)
3. [Kaggle](https://www.kaggle.com/kernels)
4. [Kaggle Introduction video](https://www.youtube.com/watch?v=FloMHMOU5Bs)
5. [Hackerrank](https://www.hackerrank.com/)
6. [Pythonanywhere](https://www.pythonanywhere.com/)

Online courses:
1. [Machine Learning A-Z: Handson using Python and R](https://www.udemy.com/share/100034B0sYd19VRXo=/)
