## Longformer for financial sentiment classification

We create a sentiment classifier for long-document financial data. 


### Architecture

The model consists of a Longformer embedding layer, and a classification head with one hidden layer and dropout. Output is a conditional distribution (in logits) over three possible label: `negative`, `neutral`, and `positive`. 


### Performance 

Currently, it achieves 72% validation accuracy and 73% validation accuracy on the FinancialPhraseBank corpus.


### Down-stream applications

The model was primarily built for a FOREX market value predictor, `FinBERT-SIMF-fx`, using market data and news sentiment as features. Previous research has focused mainly on using sentiment of news *titles* as features, while we extend it to (long sequence) article bodies as well.  


### Run the training pipeline

Due to limitation in computational resources, we run the training pipeline in a Colab notebook. Refer to `longformer-sentiment.ipynb` for more details.

### Repo structure

- `models` contains trained model checkpoints, model and training configs, and test scores. 
- `logs` contains TensorBoard logs
- `data` contains financial data used for the down-stream task referred to above.
- `exploration` contains exploratory notebooks.

