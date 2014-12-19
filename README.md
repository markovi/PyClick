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
Coming soon...