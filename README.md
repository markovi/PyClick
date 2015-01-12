Click Models for Web and Aggregated Search
====

## General Information
The code currently implements the following click models:
* **S(imple)-DCM**: Guo, Fan and Liu, Chao and Wang, Yi Min. **Efficient multiple-click models in web search.** In *Proceedings of WSDM*, pages 124-131, 2009. The parameter estimation procedure as in Equations 13 and 14: only documents above the last clicked rank are considered.
* **DCM**: The same as above, but all documents are considered when estimating parameters.
* **S(imple)-DBN**: Chapelle, Olivier and Zhang, Ya. **A dynamic bayesian network click model for web search ranking.** In *Proceedings of WWW*, pages 1-10, 2009. Parameters are estimated as in Algorithm 1 (gamma = 1).
* **DBN**: The same as above with the probability of continuation gamma.
* **UBM**: Dupret, Georges E. and Piwowarski, Benjamin. **A user browsing model to predict search engine click data from past observations.** In *Proceedings of SIGIR*, pages 331-338, 2008.
* **FCM**: Chen, Danqi and Chen, Weizhu and Wang, Haixun and Chen, Zheng and Yang, Qiang. **Beyond ten blue links: enabling user click modeling in federated web search.** In *Proceedings of WSDM*, pages 463-472, 2012.
* **VCM**: Wang, Chao and Liu, Yiqun and Zhang, Min and Ma, Shaoping and Zheng, Meihong and Qian, Jing and Zhang, Kuo. **Incorporating vertical results into search click models.** In *Proceedings of SIGIR*, pages 503-512, 2013.

Each model is implemented in two modifications:
* A basic model, where relevance is learnt from query-logs.
* A model with the "Rel" suffix (e.g., UBMRel), where relevance judgements are used to determine the relevance grade for each query-document pair. The probability of relevance for each grade is then learnt from query-logs.


## How to Use

#### Installation
To install PyClick, run the following command from the PyClick root:
 ```python setup.py install```

#### Examples
Examples of PyClick usage are in the ```examples``` folder, with the corresponding data located in ```examples/data```. To run an example ```X.py```, use the following command:
```python X.py train_data test_data```
or
```./X.py train_data test_data```
In the latter case, you may need to change permissions using ```chmod u+x X.py```.

#### SimpleExample.py
```SimpleExample.py``` takes train and test files as an input, trains several click models, outputs the trained parameters and calculates the log-likelihood, perplexity and position perplexity using test data.

To run ```SimpleExample.py```, use the following:
```python SimpleExample.py data/oneQueryTrain oneQueryTest```
or
```./SimpleExample.py data/oneQueryTrain oneQueryTest```

The train/test file format is:
```query_id doc1_id,doc2_id,...,docn_id : clicked_doc_id1,clicked_doc_id2,...;```
