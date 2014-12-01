Click Models for Web and Vertical Search
====

## General Information
The code currently implements the following click models:
* **S(imple)-DCM**: Guo, Fan and Liu, Chao and Wang, Yi Min. **Efficient multiple-click models in web search.** In *Proceedings of WSDM*, pages 124-131, 2009. The parameter estimation procedure as in Equations 13 and 14: only documents above the last clicked rank are considered.
* **DCM**: The same as above, but all documents are considered when estimating parameters.
* **S(imple)-DBN**: Chapelle, Olivier and Zhang, Ya. **A dynamic bayesian network click model for web search ranking.** In *Proceedings of WWW*, pages 1-10, 2009. Parameters are estimated as in Algorithm 1 (gamma = 1).
* **DBN**: The same as above with the probability of continuation gamma.
* **UBM**: Dupret, Georges E. and Piwowarski, Benjamin. **A user browsing model to predict search engine click data from past observations.** In *Proceedings of SIGIR*, pages 331-338, 2008.
* **JVM**: Chen, Danqi and Chen, Weizhu and Wang, Haixun and Chen, Zheng and Yang, Qiang. **Beyond ten blue links: enabling user click modeling in federated web search.** In *Proceedings of WSDM*, pages 463-472, 2012.
* **VCM**: Wang, Chao and Liu, Yiqun and Zhang, Min and Ma, Shaoping and Zheng, Meihong and Qian, Jing and Zhang, Kuo. **Incorporating vertical results into search click models.** In *Proceedings of SIGIR*, pages 503-512, 2013.

Each model is implemented in two modifications:
* A basic model, where relevance is learnt from query-logs.
* A model with the "Rel" suffix (e.g. UBMRel), where relevance judgements are used to determine the relevance grade for each query-document pair. The probability of relevance for each grade is then learnt from query-logs.


## How to Use
To run the toy example, use the following commands:
```
cd $BASE_DIR/markil
mkdir output
chmod u+x click_models.py
./click\_models.py data/train\_files data/test\_files "" test output
```
```$BASE_DIR``` is the directory where the code is located. The script trains a set of click models and outputs their parameters. Then it tests the models and compares their log-likelihood and perplexity. The trained parameters are output to ```output/params```, while the results of testing are output to ```output/test```.

Note that training can be run only once, because trained models are serialized to the ```output/models/``` directory and can be loaded from there later. To use serialized models, run the following command:
```
./click\_models.py "" data/test\_files output/models test output
```
In this case, train data is not used, while models are loaded from ```output/models/```.


#### Script Parameters
```click_models.py``` requires 5 parameters:
* A file with a list of user sessions' streams for training click models. If specified, models are trained using provided train sessions. If an empty string is passed as an argument, models are loaded from a models' serialization directory (see parameter 3).
* A file with a list of user sessions' streams for testing click models.
* A path to serialized models. This path is used to load models, if the "train file" argument is left empty (see parameter 1).
* An identifier of a set of models to be run.
* An output path.

The toy data can be found in ```data```:
* ```data/train_data``` contains user sessions for training.
* ```data/train_files``` contains a list of files/streams with training sessions. In particular, it contains the path to ```data/train_data```.
* ```data/test_data``` contains user sessions for testing.
* ```data/test_files``` contains a list of files/streams with test sessions. In particular, it contains the path to ```data/test_data```.

A list of identifiers of model sets can be found and edited in ```click_models.py```. Currently the following sets of models are available:
* **test**: S-DCM, DCM, S-DBN, DBN, UBM, JVM, VCM
* **test-rel**: S-DCMRel, DCMRel, S-DBNRel, DBNRel, UBMRel, JVMRel, VCMRel


#### User Session Format
**Note that user sessions should be in the following format:**
uid query region \[urls\] \[clicks\] \[relevance\] \[click\_times\] vertical\_position vertical\_click vertical\_click_time

* uid - a user id. Not used at the moment.
* query - a unique query identifier (usually should be unique for each query-region pair).
* region - a user region. Not used at the moment (should be considered when creating query identifiers).
* \[urls\] - a list of URLs presented to a user within a session.
* \[clicks\] - a list of clicks (0 or 1).
* \[relevance\] - the relevance of each URL, if available. -1 otherwise.
* \[click_times\] - the timestamp of each click. -1 for no-click. Not used at the moment.
* vertical_position - the rank at which a vertical result is presented.
* vertical_click - a click on a vertical (0 or 1).
* vertical\_click_time - the timestamp of a click. -1 for no-click. Not used at the moment.

