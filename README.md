---
title: "AI-900: Azure AI Fundamentals"
markmap:
  colorFreezeLevel: 3
---

## Links

- [exampro](https://app.exampro.co/student/journey/AI-900)

## Key words

- **Ground truth** is the term that describes real word
  data used to train and test AI model outputs. 
  - **Model training**: Ground truth data is used as training data, 
    where the algorithm learns which features and solutions are 
    appropriate for the specific application
  - **Model testing**: Ground truth data is used as test data, where 
    the trained algorithm is tested for model accuracy
- Learning methods
  - **Supervised learning**: The algorithm is trained on a labeled dataset, 
    where each example is a pair consisting of an input object and a 
    desired output value.
    - **Classification**: The algorithm learns to classify data into 
        different categories
    - **Regression**: The algorithm learns to predict a continuous value
  - **Unsupervised learning**: The algorithm is trained on an unlabeled dataset, 
    where the algorithm learns the patterns and relationships in the data.
    - **Clustering**: The algorithm learns to group data points into clusters
    - **Association**: The algorithm learns to discover rules that describe 
      large portions of the data
    - **Dimensionality reduction**: The algorithm learns to reduce the 
      number of random variables under consideration
  - **Reinforcement learning**: The algorithm learns to perform an action 
    from experience. It learns by trial and error, and receives feedback 
    in the form of rewards or penalties.
- **Neural network**: A neural network is a series of algorithms that 
  endeavors to recognize underlying relationships in a set of data 
  through a process that mimics the way the human brain operates.
  - **Deep learning**: A subset of machine learning that uses neural 
    networks with many layers. Deep learning algorithms can learn from 
    data that is unstructured or unlabeled. Deep learning term is used
    to describe neural networks with more than 3 hidden layer (which is
    mostly non-humanly interpretable).
  - **Feed forward neural network**: (FNN) A feedforward neural network is an 
    artificial neural network wherein connections between the nodes do 
    not form a cycle.
  - **Backpropagation**: (BP) A method used in artificial neural networks to 
    calculate a gradient that is needed in the calculation of the weights 
    to be used in the network. This is what allows the network to learn.
  - **Loss function**: A loss function is a method of evaluating how well 
    your algorithm models your dataset by comparing prediction with the 
    *ground truth* If your predictions are totally off, your loss function 
    will output a higher number. If they’re pretty good, it’ll output a 
    lower number.
  - **Activation function**: An activation function is a function that
    decides whether a neuron should be activated or not. It takes in the 
    weighted sum of the inputs from the previous layer and the bias.
- Forecasting vs. Prediction
  - **Forecasting**: Predicting future values based on historical data. Trends, 
    seasonality, and other factors are considered in forecasting.
  - **Prediction**: Predicting an outcome based without relevant data. 
    Predictions are based on patterns and relationships in the data, decision theory and is more *guessing* than forecasting.
- **Regression**
  - Correlate a labeled dataset to predict a continuous value (predict the variable in the future).
  - ![Regression line](https://upload.wikimedia.org/wikipedia/commons/3/3a/Linear_regression.svg)
    - Distance from the regression line is the **error**.
    - Multiple regression algorithms are available, such as MSE, MAE, RMSE, etc.
- **Classification**
  - Finding a function to divide a labeled dataset into categories.
  - Classification algorithms include logistic regression, decision trees, random forests, Neural Networks, Naive Bayes, etc.
- **Clustering**
  - Grouping data points into clusters based on their similarities and differences.
  - ![Single-linkage](https://upload.wikimedia.org/wikipedia/commons/c/c8/Cluster-2.svg)
  - Clustering algorithms include K-means, DBSCAN, Hierarchical clustering, etc.
- **Confusion matrix**
  - A confusion matrix is a table that is often used to describe the performance of a classification model on a set of data for which the true values are known (ground truth).
  - ![Confusion matrix](https://upload.wikimedia.org/wikipedia/commons/9/94/Contingency_table.png)
- **Anomaly detection**
  - Anomaly detection is the identification of rare items, events or observations which raise suspicions by differing significantly from the majority of the data.
  - Uses cases include fraud detection, network security, system health monitoring, etc.
- **Knowledge mining**
  - Knowledge mining is the process of extracting insights from unstructured data.
  - Uses cases include extracting information from documents, images, and other unstructured data.
  - The process includes the following steps
    1. **Ingest**: Import data from various sources.
    2. **Enrich**: Extract information from the data (cognitive services).
    3. **Explore**: Data exploration and visualization.

## Common AI workloads

### Computer vision

- Algorithms
  - **Convolutional Neural Networks** (CNN): for image and video classification, object detection, and image segmentation.
  - **Recurrent Neural Networks** (RNN): for handwriting recognition, optical character recognition (OCR), speech recognition.
- Types
  - **Image classification**: Assigns a label to an image.
  - **Object detection**: Identifies and locates objects within an image.
  - **Semantic segmentation**: Identifies the boundaries of objects within an image.
  - **Image analysis**: Extracts information from images.
  - **Optical character recognition (OCR)**: Converts images of text into machine-encoded text.
  - **Facial recognition**: Identifies or verifies a person from a digital image or video frame. Label expressions, age etc.
- Azure's Computer Vision services
  - ==Computer Vision==: Extracts information from images.
  - ==Custom Vision==: Custom image classification or object detection models.
  - ==Face==: Detects and recognizes human faces, emotions, etc.
  - ==Form Recognizer==: Extracts text, key-value pairs, and tables from documents.

### Natural Language Processing (NLP)

- Understanding the context of a corpus of text, including
  - Analyse and interpret text documents.
  - Determines the sentiment of text based on the emotions expressed.
  - Synthesizes speech
  - Translates languages
  - Determines the intent of a user's input: chatbots, virtual assistants, etc.
- Algorithms
  - **Recurrent Neural Networks** (RNN): for sequence prediction tasks.
  - **Long Short-Term Memory** (LSTM): for sequence prediction tasks.
  - **Transformer**: for sequence-to-sequence tasks.
- Azure's NLP services
  - ==Text Analytics==: Extracts insights from text.
  - ==Language Understanding==: Builds natural language understanding
  into apps, bots, and IoT devices.
  - ==Translator==: Real time text language translation.
  - ==Speech==: Converts spoken language into text.

### Conversational AI

- Conversational AI is a set of technologies that enable computers to understand, process, and respond to human language in a natural way.
  - Chatbots
  - Virtual assistants
  - Interactive Voice recognition (IVR)
- Azure's conversational AI services
  - ==QnA Maker==: Create a question-and-answer layer over your data.
  - ==Bot Services==: Build, connect, deploy, and manage intelligent bots.

## Responsible AI

### Microsoft's AI principles

1. **Fairness**: AI systems should treat all people fairly.
    - **Fairlearn**: An Open-source toolkit for assessing and improving the fairness of machine learning models.
2. **Reliability and safety**: AI systems should perform reliably and safely.
    - Test the model
    - Risks and harm related information should be accessible from the model users
3. **Privacy and security**: AI systems should respect privacy and maintain security.
    - Personally identifiable information (PII) should be protected
4. **Inclusiveness**: AI systems should empower everyone and engage people.
5. **Transparency**: AI systems should be transparent and understandable.
    - *Interpretability/Intellegibility*: The ability to explain the results of a model in a way that is understandable to humans.
6. **Accountability**: AI systems should be accountable to people.
    - **Model governance**: The process of managing the entire lifecycle of a model, including model creation, deployment, and monitoring.
    - **Organizational principles**: Define the roles and responsibilities of the people involved in the model lifecycle.

- [Guidelines for Human-AI Interaction](https://www.microsoft.com/en-us/research/project/guidelines-for-human-ai-interaction/) <!-- markmap: fold -->
  - ![](https://www.microsoft.com/en-us/research/uploads/prod/2020/03/Guidelines_summary_image@2x.png)

## Azure Cognitive Services

- Azure Cognitive Services are a set of cloud-based services that enable developers to build AI applications.
- Deploy on cloud or at the edge.
- Categories
  - Vision:
    - **Computer Vision**: Extracts information from images.
      - OCR (Optical Character Recognition)
        - ==OCR API==: older models with synchronous processing.
          - <span style="color: green">➕ Easier to implement</span>
          - <span style="color: green">➕ More language support
          - <span style="color: orange">➖ Images only</span>
          - Suitable for small documents
        - ==Read API==: newer models with asynchronous processing.
          - <span style="color: orange">➖ More difficult to implement</span>
          - <span style="color: orange">➖ Less language support</span>
          - <span style="color: green">➕ Images and PDFs</span>
          - Suitable for large documents
    - **Custom Vision**: Custom image classification or object detection models.
    - **Face**: Detects and recognizes human faces, emotions, attributes etc.
      - Provide structured data about the recognized face and related objects.
    - **Form Recognizer**: Extracts text, key-value pairs, and tables from documents.
      - Preserve the structure of the document.
      - Outuput is structured data with relationships between elements, confidence scores, etc.
      - Custom models can be trained.
        - **Supervised learning**: Provide labeled data for values of interest.
        - **Unsupervised learning**: Provide unlabeled data, understand the structure and fields.
      - Pre-built models are available: IDs, invoices, receipts, business cards, etc.
  - Speech:
    - **Speech to Text**: Converts spoken language into text.
    - **Text to Speech**: Converts text into spoken language.
    - **Speech translation**: Real-time translation of spoken language.
    - **Speaker recognition**: Uses voice characteristics to identify a person.
  - Language:
    - **Text Analytics**: Extracts insights from text.
      - **Sentiment Analysis**: Determines the sentiment of text based on the emotions expressed.
        - Confidence score between 0 and 1.
      - **Key Phrase Extraction**: Identifies the main points in text.
      - **Opinion Mining**: Identifies opinions in text.
      - **Language Detection**: Identifies the language of text.
      - **Named Entity Recognition**: (NER) Identifies entities in text.
      - **Key Phrase Extraction**: Identifies the main points in text.
    - **Language Understanding**: Builds natural language understanding into apps, bots, and IoT devices.
      - [**LUIS**](https://luis.ai): Language Understanding Intelligent Service.
        - No-code ML service to build natural language understanding into apps, bots, and IoT devices.
        - Luis is based on NLP (Natural Language Processing) and NLU (Natural Language Understanding).
        - Luis uses intents and entities extraction to understand the user's input.
        - **Utterances** are the input sentences that the model will learn from.
        - Luis will be retired on October 1, 2025 and is replaced by Conversation Language Understanding (CLU) in ==Language Studio==.
      - [Conversation Language Understanding](https://language.cognitive.azure.com/clu/projects)
    - **Translator**: Real-time text language translation.
    - [**QnA Maker**](https://www.qnamaker.ai/): Create a question-and-answer layer over your data.
      - NLP service to create a conversation/question-and-answer layer over your data.
      - Uses a fixed knowledge base to answer questions with repetable answers.
      - Data is stored in Azure Search, then QnA Maker uses the top results to answer questions with a confidence score.
      - QnAMaker is now deprecated and Custom Question and Answering is part of ==Language Studio==.
  - Decision:
    - **Anomaly Detector**: Detect anomalies in time series data.
    - **Personalizer**: Provide personalized content in real time.
    - **Content Moderator**: Detect potential offensive content.
- Access to Azure Cognitive Services is provided by an *API Endpoint* and an *API Key*.
