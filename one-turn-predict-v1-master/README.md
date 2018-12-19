## one-turn-predict-v1

1. Vanilla seq2seq
2. Seq2seq + Attention (Luong/Bahdanau)
3. Word embeddings + Attention (Luong/Bahdanau)

#### Execution steps:

* Build vocabulary:
  * cd vanilla_seq2seq / seq2seq_with_attentin / word_embed_attention 
  * python helper_scripts/build_vocabulary.py 
      * update training file
      * generates and stores vocab (word2id, id2word dicts) files    
* Update config settings in train.py
* Train the model: 
  * python train.py
* Test the model:
  * python cli_decode.py

##### Note: 
 * Beam search is not completely implemented. Will update the section later.

### Attention on the word embeddings (experiment-3)
![architecture](https://media.github.home.247-inc.net/user/732/files/e5f6051c-6b02-11e8-9128-8466bd74a387)

#### Results (@76th Epoch):
| Experiment | Train Perplexity | Valid Perplexity |
| :--- | :---: | :---: |
| Vanilla seq2seq | 4.16 | 4.17 |
| Seq2seq + Attention | 3.69 | 3.81 |
| Word embeddings + Attention | 3.78 | 3.82 |
