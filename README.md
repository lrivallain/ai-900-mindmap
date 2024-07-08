---
title: "AI-900: Azure AI Fundamentals"
markmap:
  colorFreezeLevel: 3
---

## Key words

- **AI**: Computer systems that can perform tasks that normally 
          require human intelligence by simulating or replicating
          human intelligence.
- **Generative AI**: Subset of AI that creates new data and content.
- **Large Language Models**: AI models that can generate human-like text.
  - Trained on large datasets
  - Understands the context
  - Can be refined from feedback
  - **Transformer model** is especially good at understanding and generating language.
  - **Tokenization**: The process of breaking text into smaller units called tokens (like a puzzle).
  - **Embedding**: The process of converting words into numeric codes.
    - Helps to understand the context of the words and the relationships or similarities between them.
  - **Positional encoding**: The process of adding information about the position 
    of the words in the text.
    - Helps the model to understand the order of the words and the meaning of the text.
  - **Attention mechanism**: The process of focusing on specific parts of the text: 
    like a light on a specific word.
    - Helps to encode the context of the words and the relationships between them.
    - Can be multi-headed to focus on different parts of the text.
    - Used to generate the output of the model, one word/token at a time.
- **Prompt engineering**: The process of creating a prompt that guides the model to 
  generate the desired output.
  - **Prompt**: A set of instructions or questions that guide the model to generate
      the desired output.
  - **Prompt design**: The process of creating a prompt that guides the model to 
    generate the desired output.
      - System messages describe expectations and constraints: 
      `"Write a *summary* of the text in 3 sentences, French language."`
  - **Prompt tuning**: The process of refining the prompt to improve the model's output.
  - Examples in prompts:
    - **Zero-shot learning**: The process of training a model to perform a task without 
      any examples.
    - **Few-shot learning**: The process of training a model to perform a task with a 
      small number of examples.
  - **Grounding**: The process of providing context to the model to generate the desired output.
    
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
      - Numeric labels/values are used and predicted
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
  - **Feed forward neural network**: (FNN) A feed forward neural network is an 
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
    Predictions are based on patterns and relationships in the data, 
    decision theory and is more *guessing* than forecasting.
- **Regression**
  - Correlate a labeled dataset to predict a continuous value (predict the variable in the future).
  - ![Regression line](https://upload.wikimedia.org/wikipedia/commons/3/3a/Linear_regression.svg)
    - Distance from the regression line is the **error**.
    - Multiple regression algorithms are available, such as MSE, MAE, RMSE, etc.
- **Classification**
  - Finding a function to divide a labeled dataset into categories.
  - Classification algorithms include logistic regression, decision trees,
    random forests, Neural Networks, Naive Bayes, etc.
- **Clustering**
  - Grouping data points into clusters based on their similarities and differences.
  - ![Single-linkage](https://upload.wikimedia.org/wikipedia/commons/c/c8/Cluster-2.svg)
  - Clustering algorithms include K-means, DBSCAN, Hierarchical clustering, etc.
- **Confusion matrix**
  - A confusion matrix is a table that is often used to describe the 
    performance of a classification model on a set of data for which 
    the true values are known (ground truth).
  - ![Confusion matrix](https://upload.wikimedia.org/wikipedia/commons/9/94/Contingency_table.png)
- **Anomaly detection**
  - Anomaly detection is the identification of rare items, events or 
    observations which raise suspicions by differing significantly from the majority of the data.
  - Uses cases include fraud detection, network security, system health monitoring, etc.
- **Knowledge mining**
  - Knowledge mining is the process of extracting insights from unstructured data.
  - Uses cases include extracting information from documents, images, and 
    other unstructured data.
  - The process includes the following steps
    1. **Ingest**: Import data from various sources.
    2. **Enrich**: Extract information from the data (cognitive services).
    3. **Explore**: Data exploration and visualization.

## Common AI workloads

### Computer vision

- Algorithms
  - **Convolutional Neural Networks** (CNN): for image and video classification, 
    object detection, and image segmentation.
  - **Recurrent Neural Networks** (RNN): for handwriting recognition, optical 
    character recognition (OCR), speech recognition.
- Types
  - **Image classification**: Assigns a label to an image.
  - **Object detection**: Identifies and locates objects within an image.
  - **Semantic segmentation**: Identifies the boundaries of objects within an image.
  - **Image analysis**: Extracts information from images.
  - **Optical character recognition (OCR)**: Converts images of text into machine-encoded text.
  - **Facial recognition**: Identifies or verifies a person from a digital 
    image or video frame. Label expressions, age etc.
- Azure's Computer Vision services
  - ==Computer Vision==: Extracts information from images.
  - ==Custom Vision==: Custom image classification or object detection models.
  - ==Face==: Detects and recognizes human faces, emotions, etc.
  - ==Form Recognizer==: Extracts text, key-value pairs, and tables from documents.

### Natural Language Processing (NLP)

- Understanding the context of a corpus of text, including
  - Analyze and interpret text documents.
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

- Conversational AI is a set of technologies that enable computers to 
  understand, process, and respond to human language in a natural way.
  - Chatbots
  - Virtual assistants
  - Interactive Voice recognition (IVR)
- Azure's conversational AI services
  - ==QnA Maker==: Create a question-and-answer layer over your data.
  - ==Bot Services==: Build, connect, deploy, and manage intelligent bots.

## Responsible AI

### Microsoft's AI principles

1. **Fairness**: AI systems should treat all people fairly.
    - **Fairlearn**: An Open-source toolkit for assessing and improving the 
      fairness of machine learning models.
2. **Reliability and safety**: AI systems should perform reliably and safely.
    - Test the model
    - Risks and harm related information should be accessible from the model users
3. **Privacy and security**: AI systems should respect privacy and maintain security.
    - Personally identifiable information (PII) should be protected
4. **Inclusiveness**: AI systems should empower everyone and engage people.
5. **Transparency**: AI systems should be transparent and understandable.
    - *Interpretability/Intellegibility*: The ability to explain the results of a model
      in a way that is understandable to humans.
6. **Accountability**: AI systems should be accountable to people.
    - **Model governance**: The process of managing the entire lifecycle of a model, 
      including model creation, deployment, and monitoring.
    - **Organizational principles**: Define the roles and responsibilities of the 
      people involved in the model lifecycle.

- [Guidelines for Human-AI Interaction](https://www.microsoft.com/en-us/research/project/guidelines-for-human-ai-interaction/) <!-- markmap: fold -->
  - ![](https://www.microsoft.com/en-us/research/uploads/prod/2020/03/Guidelines_summary_image@2x.png)

## Azure Cognitive Services

- Azure Cognitive Services are a set of cloud-based services 
  that enable developers to build AI applications.
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
        - Luis will be retired on October 1, 2025 and is replaced by 
          Conversation Language Understanding (CLU) in ==Language Studio==.
      - [Conversation Language Understanding](https://language.cognitive.azure.com/clu/projects)
    - **Translator**: Real-time text language translation.
    - [**QnA Maker**](https://www.qnamaker.ai/): Create a question-and-answer layer over your data.
      - NLP service to create a conversation/question-and-answer layer over your data.
      - Uses a fixed knowledge base to answer questions with repeatable answers.
      - Data is stored in Azure Search, then QnA Maker uses the top results to 
      answer questions with a confidence score.
      - Multi-turn conversations are supported (A question can be followed by another 
      question to clarify the answer).
      - Can suggest changes to the knowledge base based on real life usage.
      - QnAMaker is now deprecated and Custom Question and Answering is part of ==Language Studio==.
  - Decision:
    - **Anomaly Detector**: Detect anomalies in time series data.
    - **Personalizer**: Provide personalized content in real time.
    - **Content Moderator**: Detect potential offensive content.
- Access to Azure Cognitive Services is provided by an *API Endpoint* and an *API Key*.

### Azure AI Bot Services

- Azure AI Bot Service provides an integrated, serverless environment that is purpose-built for bot 
development, publishing and managing.
- ==Bot Framework SDK== is an open-source SDK for building and connecting intelligent bots.
  - Works along with Azure Bot Service.
- ==Bot Framework Composer== is a visual authoring canvas for developers and 
    multi-disciplinary teams to design and build conversational experiences with Language 
    Understanding (LUIS) and QnA Maker.
  - Deploy bots on Azure Web Apps, Azure Functions
  - Template for common scenarios: QnA, Personal Assistant, Language bot, etc.
  - Bot Framework Emulator for testing.

### Azure Machine Learning Service

- Products
  - ==Azure Machine Learning Studio (classic)== is the legacy service to create AI/ML workloads: not 
  easily migrated to newer services.
  - ==Azure Machine Learning Service== is a cloud-based environment you can use to train, deploy,
  automate, manage, and track flexible ML models. Use Python, R, TensorFlow, PyTorch, etc.
    - **Jupiter notebooks**: Build, share, collaborate, and run ML models
    - **Azure Machine Learning SDK**: Python SDK to interact with Azure Machine Learning service.
    - **Azure Machine Learning Designer**: Drag-and-drop interface to create and deploy ML models.
    - **ML Ops**: DevOps for ML, CI/CD, model versioning, etc.
- Activities
  - Labeling
    - **Human in the loop**: A process where a human is involved in the decision-making process.
    - **Machine learning assisted labeling**: A process where a machine learning model is used 
    to assist in the labeling process.
    - Export
      - COCO format
      - Azure Machine Learning dataset
  - Data Stores
    - Azure Blob storage
    - Azure File Share
    - Azure Data Lake Storage (Gen1 and Gen2)
    - Azure SQL Database
    - Azure PostgreSQL Database
    - Azure MySQL Database
  - Datasets
    - Store and version datasets
    - Profile: Summary statistics and distribution of the dataset
    - Open datasets: Azure curated list of public datasets available for use
  - Experiments
    - Logical grouping of runs
    - Runs: A single execution of an ML task
  - Pipeline
    - Executable workflow of an ML task
    - Steps: Individual components of a pipeline
    - Allow data scientists to work at the same time on different steps
  - Inference pipelines
    - Real time inference
    - Batch inference
  - Models
    - Register, deploy and monitor
    - Versioning
  - Endpoints
    - Deploy models as a web service
    - Serverless API endpoints, online endpoints, and batch endpoints
  - Notebooks
    - Built-in Jupyter notebook editor
    - Python, R, and Scala

### Auto ML

- Automated machine learning (AutoML) is the process of automating the time-consuming, 
  iterative tasks of machine learning model development. It allows data scientists, analysts, 
  and developers to build ML models with high scale, efficiency, and productivity.

- **Classification**: Supervised learning where the target variable is categorical.
  - **Binary classification**: Predicts one of two classes.
  - **Multi-class classification**: Predicts one of multiple classes.
- **Regression**: Supervised learning where to predict a variable in the future.
- **Time series forecasting**: Predicts future values based on historical data.
  - Can be considered as a multivariate *regression*
- **Automatic Featurization**: Checks the data for common issues that can 
  cause the model to perform poorly (Data guardrails).
  - Apply the appropriate transformation to the data: Scaling or normalization.
- **Model selection**: Selects the best candidate model based on the data and the task.
- **ML Explainability**: Understand the model's decision-making process.
- **Primary metric**: The metric used to evaluate and optimize the model during training.
  - Set of metrics for scenarios of classification, regression, and time series forecasting.
- **Validation**: The process of evaluating the model's performance on unseen data.

### [Custom Vision](https://customvision.ai/)

- Custom Vision is a cloud-based (no-code / managed) service that enables 
  you to build, deploy, and improve your own image classifiers or object detection models.
  1. Upload images
  2. Train the model on uploaded images
  3. Evaluate the model on images
- Project types:
  - **Classification**: Assigns a label to an image.
    - **Multiclass**: Assigns a single label to an image.
    - **Multilabel**: Assigns multiple labels to an image.
    - When training is completed, report is generated with following metrics:
      - Precision (Relevance)
      - Recall (Sensitivity)
      - Average Precision (AP)
    - Smart Labeler: Suggests labels based on the uploaded images.
  - **Object detection**: Identifies and locates objects within an image.
    - When training is completed, report is generated with following metrics:
      - Precision (Relevance)
      - Recall (Sensitivity)
      - Mean Average Precision (mAP)
  - **Domain**: Select a Microsoft managed datasets to train the model.
- Model publication is only available in the same region as the training.
  - **Prediction URL** is provided to use the model in your application.

## Azure OpenAI Services

- **OpenAI** is an American artificial intelligence (AI) research organization founded 
  in December 2015, researching artificial intelligence with the goal of developing 
  *"safe and beneficial"* artificial general intelligence.
- **OpenAI Services** is a cloud-based platform to deploy and manage AI models from OpenAI.
  - ==GPT-3.5==: A large language model that can generate human-like text in conversational style.
  - ==GPT-4==: A large language model that can generate text and code based on natural language inputs.
  - ==Embeddings models==: A service that can generate numeric codes for words and 
    sentences for comparison.
  - ==DALL-E==: A model that can generate images from textual descriptions.
- ==Azure OpenAI Studio==: Web-based interface to interact with OpenAI services.
- Pricing is based on the number of tokens used to interact with the model (input) 
  and to generate the output.
    - Depending on the used model, context is limited to a certain number of tokens.

## Copilots

- Type of tools that helps users with common tasks using generative AI.
  - Trained LLM with a large dataset of data.
  - Can use Azure OpenAI services to be fine-tuned on specific tasks and data.
  - Increases productivity by providing suggestions, synthesis, planning and more.
- Examples:
  - ==GitHub Copilot==: An AI-powered code completion tool that suggests real time code snippets 
    based on the context of the code.
  - ==M365 Copilot==: An AI-powered tool that helps users with common tasks in 
    Microsoft Office applications.
  - ==Bing Copilot==: An AI-powered tool that helps users in Bing search with natural 
    language queries.
  