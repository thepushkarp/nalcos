---
pipeline_tag: sentence-similarity
tags:
- sentence-transformers
- feature-extraction
- sentence-similarity
---

# multi-qa-MiniLM-L6-cos-v1
This is a [sentence-transformers](https://www.SBERT.net) model: It maps sentences & paragraphs to a 384 dimensional dense vector space and was designed for **semantic search**. It has been trained on 215M (question, answer) pairs from diverse sources. For an introduction to semantic search, have a look at: [SBERT.net - Semantic Search](https://www.sbert.net/examples/applications/semantic-search/README.html)


## Usage (Sentence-Transformers)
Using this model becomes easy when you have [sentence-transformers](https://www.SBERT.net) installed:

```
pip install -U sentence-transformers
```

Then you can use the model like this:
```python
from sentence_transformers import SentenceTransformer, util

query = "How many people live in London?"
docs = ["Around 9 Million people live in London", "London is known for its financial district"]

#Load the model
model = SentenceTransformer('sentence-transformers/multi-qa-MiniLM-L6-cos-v1')

#Encode query and documents
query_emb = model.encode(query)
doc_emb = model.encode(docs)

#Compute dot score between query and all document embeddings
scores = util.dot_score(query_emb, doc_emb)[0].cpu().tolist()

#Combine docs & scores
doc_score_pairs = list(zip(docs, scores))

#Sort by decreasing score
doc_score_pairs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)

#Output passages & scores
for doc, score in doc_score_pairs:
    print(score, doc)
```


## Usage (HuggingFace Transformers)
Without [sentence-transformers](https://www.SBERT.net), you can use the model like this: First, you pass your input through the transformer model, then you have to apply the correct pooling-operation on-top of the contextualized word embeddings.

```python
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

#Mean Pooling - Take average of all tokens
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


#Encode text
def encode(texts):
    # Tokenize sentences
    encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')

    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input, return_dict=True)

    # Perform pooling
    embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

    # Normalize embeddings
    embeddings = F.normalize(embeddings, p=2, dim=1)
	
    return embeddings


# Sentences we want sentence embeddings for
query = "How many people live in London?"
docs = ["Around 9 Million people live in London", "London is known for its financial district"]

# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/multi-qa-MiniLM-L6-cos-v1")
model = AutoModel.from_pretrained("sentence-transformers/multi-qa-MiniLM-L6-cos-v1")

#Encode query and docs
query_emb = encode(query)
doc_emb = encode(docs)

#Compute dot score between query and all document embeddings
scores = torch.mm(query_emb, doc_emb.transpose(0, 1))[0].cpu().tolist()

#Combine docs & scores
doc_score_pairs = list(zip(docs, scores))

#Sort by decreasing score
doc_score_pairs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)

#Output passages & scores
for doc, score in doc_score_pairs:
    print(score, doc)
```

## Technical Details

In the following some technical details how this model must be used:

| Setting | Value |
| --- | :---: |
| Dimensions | 384 |
| Produces normalized embeddings | Yes |
| Pooling-Method | Mean pooling |
| Suitable score functions | dot-product (`util.dot_score`), cosine-similarity (`util.cos_sim`), or euclidean distance |

Note: When loaded with `sentence-transformers`, this model produces normalized embeddings with length 1. In that case, dot-product and cosine-similarity are equivalent. dot-product is preferred as it is faster. Euclidean distance is proportional to dot-product and can also be used.

----


## Background

The project aims to train sentence embedding models on very large sentence level datasets using a self-supervised 
contrastive learning objective. We use a contrastive learning objective: given a sentence from the pair, the model should predict which out of a set of randomly sampled other sentences, was actually paired with it in our dataset.

We developped this model during the 
[Community week using JAX/Flax for NLP & CV](https://discuss.huggingface.co/t/open-to-the-community-community-week-using-jax-flax-for-nlp-cv/7104), 
organized by Hugging Face. We developped this model as part of the project:
[Train the Best Sentence Embedding Model Ever with 1B Training Pairs](https://discuss.huggingface.co/t/train-the-best-sentence-embedding-model-ever-with-1b-training-pairs/7354). We benefited from efficient hardware infrastructure to run the project: 7 TPUs v3-8, as well as intervention from Googles Flax, JAX, and Cloud team member about efficient deep learning frameworks.

## Intended uses

Our model is intented to be used for semantic search: It encodes queries / questions and text paragraphs in a dense vector space. It finds relevant documents for the given passages.

Note that there is a limit of 512 word pieces: Text longer than that will be truncated. Further note that the model was just trained on input text up to 250 word pieces. It might not work well for longer text. 



## Training procedure

The full training script is accessible in this current repository: `train_script.py`.

### Pre-training 

We use the pretrained [`nreimers/MiniLM-L6-H384-uncased`](https://huggingface.co/nreimers/MiniLM-L6-H384-uncased) model. Please refer to the model card for more detailed information about the pre-training procedure.

#### Training

We use the concatenation from multiple datasets to fine-tune our model. In total we have about 215M (question, answer) pairs.
We sampled each dataset given a weighted probability which configuration is detailed in the `data_config.json` file.

The model was trained with [MultipleNegativesRankingLoss](https://www.sbert.net/docs/package_reference/losses.html#multiplenegativesrankingloss) using Mean-pooling, cosine-similarity as similarity function, and a scale of 20.




| Dataset                    | Number of training tuples  |
|--------------------------------------------------------|:--------------------------:|
| [WikiAnswers](https://github.com/afader/oqa#wikianswers-corpus) Duplicate question pairs from WikiAnswers |  77,427,422 |
| [PAQ](https://github.com/facebookresearch/PAQ) Automatically generated (Question, Paragraph) pairs for each paragraph in Wikipedia | 64,371,441 |
| [Stack Exchange](https://huggingface.co/datasets/flax-sentence-embeddings/stackexchange_xml) (Title, Body) pairs from all StackExchanges  | 25,316,456 |
| [Stack Exchange](https://huggingface.co/datasets/flax-sentence-embeddings/stackexchange_xml) (Title, Answer) pairs from all StackExchanges  |  21,396,559 |
| [MS MARCO](https://microsoft.github.io/msmarco/) Triplets (query, answer, hard_negative) for 500k queries from Bing search engine |  17,579,773 |
| [GOOAQ: Open Question Answering with Diverse Answer Types](https://github.com/allenai/gooaq) (query, answer) pairs for 3M Google queries and Google featured snippet  | 3,012,496 |
| [Amazon-QA](http://jmcauley.ucsd.edu/data/amazon/qa/) (Question, Answer) pairs from Amazon product pages | 2,448,839 
| [Yahoo Answers](https://www.kaggle.com/soumikrakshit/yahoo-answers-dataset) (Title, Answer) pairs from Yahoo Answers | 1,198,260 |
| [Yahoo Answers](https://www.kaggle.com/soumikrakshit/yahoo-answers-dataset) (Question, Answer) pairs from Yahoo Answers | 681,164 |
| [Yahoo Answers](https://www.kaggle.com/soumikrakshit/yahoo-answers-dataset) (Title, Question) pairs from Yahoo Answers | 659,896 |
| [SearchQA](https://huggingface.co/datasets/search_qa) (Question, Answer) pairs for 140k questions, each with Top5 Google snippets on that question | 582,261 |
| [ELI5](https://huggingface.co/datasets/eli5) (Question, Answer) pairs from Reddit ELI5 (explainlikeimfive) | 325,475 |
| [Stack Exchange](https://huggingface.co/datasets/flax-sentence-embeddings/stackexchange_xml) Duplicate questions pairs (titles) | 304,525 |
| [Quora Question Triplets](https://quoradata.quora.com/First-Quora-Dataset-Release-Question-Pairs) (Question, Duplicate_Question, Hard_Negative) triplets for Quora Questions Pairs dataset | 103,663 |
| [Natural Questions (NQ)](https://ai.google.com/research/NaturalQuestions) (Question, Paragraph) pairs for 100k real Google queries with relevant Wikipedia paragraph | 100,231 |
| [SQuAD2.0](https://rajpurkar.github.io/SQuAD-explorer/) (Question, Paragraph) pairs from SQuAD2.0 dataset |  87,599 |
| [TriviaQA](https://huggingface.co/datasets/trivia_qa) (Question, Evidence) pairs | 73,346 |
| **Total** | **214,988,242** |