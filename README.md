# Medical Entity Extraction from Clinical Notes

## Collaborators
[Deep C. Patel](https://github.com/deepcpatel) and [Akshay Shah](https://github.com/shahakshay11)

## Dataset used
2010 i2b2/VA challenge dataset [5]. 

## Background
The 2010 i2b2/VA challenge task 1 [5] is basically to extract medical entities from unannotated clinical texts. This task can be categorized under the umbrella of classic NLP task known as Named Entity Recognition. Here, the extracted medical entities can be classified into a medical problem, a treatment, or a test. We are attempting to solve this task using Differentiable Neural Computer [1].

## Our approach
Inspiring from the usage of Differentiable Neural Computer (DNC) for the question answering Task [1, 2], we tried to use it for the Medical Entity Extraction task in this course project. The advantage of using it over LSTMs is that, unlike them, DNC keeps Neural Network and Memory separate or in other words Neural Network controller is attached to external memory, which allows larger memory allocation without an increase in the parameters. Moreover, the stored information in the memory faces very little interference from the network operations, thus allowing it to be remembered for a longer time.
Additionally, we finetuned the MIMIC-III[6] pretrained BERT model on the clinical text for the entity recognition task using Huggingface transformers library.


## Adapting DNC for Medical Entity Extraction task
As indicated above, DNC was used for question answering task in the original paper [1]. However, it was not performing robustly as pointed out by the authors in [2]. The prime issue suspected was that the DNC might not be using memory units in some cases (not using read vectors) and instead be solely generating output using only the given inputs. This issue was solved by putting the Dropout layer before the final output generation, which allowed both the components to be used equally. Moreover, the authors in [2] perform Layer Norm on the controller output to control high variance in performance during different DNC runs.

To get the information of future tokens in the sequence, the authors in [2] also proposed to use a backward LSTM controller in parallel to DNC. These modifications to DNC showed robust performance during question answering task according to them. Thus, to make our Entity Extraction task robust, we also modified our DNC implementation as recommended in [2].

We approached the Medical Entity Extraction task as a token classification task; the classes being `Problem`, `Treatment` and `Test`. And since we adopted BIO tagging as our labeling convention, specifically our classes are `B-Problem`, `I-Problem`, `B-Test`, `I-Test`, `B-Treatment`, `I-Treatment` and `Other`. Finally, we implemented DNC in PyTorch and programmed it to classify each input tokens into one of these classes.

## Data pre-processing
* DNC </br>
As such, the data required a moderate amount of preprocessing. We first tokenized each Medical Summary into words and removed noisy characters such as `\n` and more than one space. From the `.con` files, we extracted entities for making true labels. After this step, we converted all the words into word2vec vectors using the pre-trained word2vec embeddings on the PubMed corpus and MeSH RDF from [3]. After that, we divided the data into batches and fed them into the DNC.
* MIMIC-BERT based Token Classifier </br>
Similar to pre processing for DNC,with only major difference in using tokenized sentences using BertTokenizer and marking the word piece tokens after BERT tokenizes it as X such that it can be discarded while checking the final tags. We feed the tokenized inputs and tags with standard test train split to train the BERT model

## Training
* DNC </br>
We trained our DNC with a batch size of 10 for 100 epochs using Nvidia GTX 1080 GPU. The training took us approximately 10 hours. To update weights, we calculated Softmax Cross-Entropy loss between predicted output and labels and propagated gradients back to the model for weight updation. Since the DNC is fully differentiable end to end, the backpropagation algorithm can be easily used to update its weights. We used only one read and write head in DNC with a memory size of 128 x 128.
* MIMIC-BERT based Token Classifier </br>
We finetune the BERT by training the model for 5 epochs with Adam Optimizer with batch size of 16 to avoid any memory errors.
To update weights, BertForTokenClassification computes Softmax Cross-Entropy loss between predicted output and labels and propagated gradients back to the model for weight updation.

## Results
* DNC </br>
The results of our model are outlined as follows. Please note that, we consider a classification to be True Positive only if all the words in an entity correctly matches with that of the corresponding label (exact match) and hence adds +1 to our True Positive score, else adds +1 to False Negative score.

| Entity Type | Precision | Recall | F1 Score |
|---|---|---|---|
| problem | 0.78 | 0.74 | 0.76 |
| test | 0.85 | 0.62 | 0.72 |
| treatment | 0.83 | 0.62 | 0.71 |

Total entity classification accuracy: **66.76 %**<br/>
Macro average F1 Score: **0.73**

* MIMIC-BERT based Token Classifier

| Entity Type | Precision | Recall | F1 Score |
|---|---|---|---|
| problem | 0.84 | 0.85 | 0.85 |
| test | 0.84 | 0.9 | 0.87 |
| treatment | 0.87 | 0.88 | 0.87 |

Total entity classification accuracy: **99.78 %**<br/>
Macro average F1 Score: **0.87**


## Future work
(1).​ Use BERT embeddings extracted from finetuned MIMIC BERT as word embeddings for the data input to DNC instead of word2vec.<br/>
(2). Along with word embeddings, add character level embeddings, Parts Of Speech information to input as outlined in [4].

## Final message
We welcome contributions, suggestions or any bug findings in our work.

## References
**[1]**. A. Graves, et. al. Hybrid Computing Using a Neural Network with Dynamic External Memory. Nature 538, 471-476, 2016.<br/>
**[2]**. ​J. Franke, J. Niehues, A. Waibel. Robust and Scalable Differentiable Neural Computer for Question Answering. arXiv preprint arXiv:1807.02658, 2018.<br/>
**[3]**. Y. Zhang, Q. Chen, Z. Yang, H. Lin, Z. Lu. ​BioWordVec, Improving biomedical word embeddings with subword information and MeSH. Scientific Data, 2019.<br/>
**[4]**. J.P.C. Chiu, E. Nichols. Named Entity Recognition with Bidirectional LSTM-CNNs. Transactions of the Association for Computational Linguistics, Volume 4, 2016.<br/>
**[5]**. Ö. Uzuner, B.R. South, S. Shen, et al. 2010 i2b2/VA challenge on concepts, assertions, and relations in clinical text. J Am Med Inform Assoc, 18:552–6, 2011.</br>
**[6]**. Peng, Yifan, Shankai Yan, and Zhiyong Lu. "Transfer Learning in Biomedical Natural Language Processing: An Evaluation of BERT and ELMo on Ten Benchmarking Datasets." arXiv preprint arXiv:1906.05474 (2019).
