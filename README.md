PyClick - Click Models for Web Search
====

PyClick is an open-source Python library of click models for web search.
It implements all standard click models and most inference methods described in the following book:

```
Aleksandr Chuklin, Ilya Markov, Maarten de Rijke.
Click Models for Web Search.
Morgan & Claypool Publishers, 2015.
http://clickmodels.weebly.com/the-book.html
```
        

# How to Use

## Installation
To install the PyClick library, run the following code in command line:

```bash
cd $PROJECT_DIR
sudo python setup.py install
```

Dependencies:

* enum34


## Running examples
It is highly recommended to use the [PyPy](http://pypy.org/) interpreter.
It speeds up the code 10-100 times.

Examples are located in the ```examples``` folder.
Data samples are in ```examples/data```.
Examples can be run as follows:

```
python examples/SimpleExample.py $CLICK_MODEL examples/data/YandexRelPredChallenge $SESSION_NUM
```

Here, ```$CLICK_MODEL``` is the click model to use for this example (see the list of implemented models below);
```$SESSION_NUM``` is the number of search sessions to consider.

Currently, the following click models are implemented and can be used for this example
(see Chapter 3 of our book [1]):

* GCTR (global CTR, aka, random click model)
* RCTR (rank-based CTR)
* DCTR (document-based CTR)
* PBM (position-based model) [2]
* CM (cascade model) [2]
* UBM (user browsing model) [3]
* DCM (dependent click model) [4]
* CCM (click-chain model) [5]
* DBN (dynamic Bayesian network) [6]
* SDBN (simplified DBN) [6]

There is a separate example for the task-centric click model (TCM) [7].


## Implementing a new click model
1. Inherit from ```pyclick.click_models.ClickModel```
  
        class NewClickModel(ClickModel)

2. Define the names of the model parameters:

        param_names = Enum('NCMParamNames', 'one_param another_param')

3. Choose appropriate containers for these parameters (see more on this below).
Usually, relevance-related parameters depend on a query and a document
and so can be stored in ```QueryDocumentParamContainer```.
Examination-related parameters usually depend on the document rank
and so can be stored in either ```RankParamContainer``` or ```RankPrevClickParamContainer```.
Sometimes, there is a single examination parameter, stored in ```SingleParamContainer```.

4. Choose an appropriate inference method for the click model (see more on this below).
If all random variables of the model are observed, use ```MLEInference```.
Otherwise, use ```EMInference```. For other options see below.

5. Initialize the click model using the chosen parameter names, containers and inference:

        def __init__(self):
            # Specific containers are used just for the purpose of example
            self.params = {self.param_names.one_param: QueryDocumentParamContainer(),
                           self.param_names.another_param: RankParamContainer.default()}
            # MLE inference is used just for the purpose of example
            self._inference = MLEInference()
            
6. Implement model parameters.
Note that the parameter implementation usually depends on the chosen inference method,
so the same model can have different implementations of its parameters
for different inference methods.
For example, the standard DBN model uses the EM inference,
while its simplified version uses the MLE inference.
These the two versions of DBN need different implementations of the DBN parameters.
Thus, it makes sense to name parameter classes as follows: <ModelName>Param<InferenceName>.
To implement a click model parameter, follow this procedure:

* Inherit from ```pyclick.click_models.Param``` or one of its children.
For example, there are predefined classes ```pyclick.click_models.ParamEM```
and ```click_models.ParamMLE``` with basic functionality for parameters
that implement either EM or MLE inference.

        class NCMParamMLE(ParamMLE)
        
* Implement the ```update``` method.
This implementation depends on the chosen inference method.
For example, in the MLE inference,
the values of parameters for a particular search result in a particular search session are calculated
based on this search session and the rank of the result.
In the EM inference, the values of parameters from the previous iteration
are used in addition to the session and rank.
For the ready-to-use updating formulas of standard click models,
please refer to Chapter 4 of our book [1].
Updating formulas for new click models
can also be derived based on instructions of Chapter 4.

7. Implement the calculation of full and conditional click probabilities.
For the ready-to-use formulas please refer to Chapter 3 of the book [1].

  * ```get_conditional_click_probs```:
  Returns a list of click probabilities conditioned on the observed clicks in the given search session.
  In particular, for a result at rank ```k``` calculates the following probability:
  ```P(C_k | C_1, C_2, ..., C_k-1)```,
  where ```C_i``` is 1 if there is a click on the ```i```-th result in the given search session and 0 otherwise.
  * ```predict_click_probs```:
  Returns a list of full click probabilities ```P(C = 1)```
  for all results in the given search session.


#### Containers of click model parameters (```ParamContainer```)

* ```QueryDocumentParamContainer```: A container of click model parameters that depend on a query-document pair.
Used in almost all standard click models for the attractiveness parameters.
* ```RankParamContainer```: A container of click model parameters that depend on rank.
Usually used to store the examination parameters (e.g., in PBM).
* ```RankPrevClickParamContainer```: A container of click model parameters that depend on rank
and on the rank of the previously clicked result.
Used only in UBM to store the examination parameters.
However, UBM is a popular model that has very many extensions (see Chapter 8 of the book [1]),
so this parameter container becomes an important one.
* ```SingleParamContainer```: A container of a click model parameter that does not depend on anything
(e.g., continuation probability in DBN).


#### Inference methods for click models (```Inference```)
TODO


# Acknowledgements

* The project is partially funded by the grant P2T1P2\_152269 of the Swiss National Science Foundation.
* Initially inspired by the [clickmodels](https://github.com/varepsilon/clickmodels) project.
* Contributors: Ilya Markov, Aleksandr Chuklin, Artem Grotov, Luka Stout, Finde Xumara, Bart Vredebregt, Nick de Wolf.


# References
[1] Aleksandr Chuklin, Ilya Markov, Maarten de Rijke.
Click Models for Web Search.
Morgan & Claypool Publishers, 2015.

[2] Nick Craswell, Onno Zoeter, Michael Taylor, and Bill Ramsey. An experimental comparison of click position-bias models. In WSDM, pages 87–94, New York, NY, USA, 2008. ACM Press. doi: 10.1145/1341531.1341545

[3] Georges E. Dupret and Benjamin Piwowarski. A user browsing model to predict search engine click data from past observations. In SIGIR, pages 331–338, New York, NY, USA, 2008. ACM Press. doi: 10.1145/1390334.1390392

[4] Fan Guo, Chao Liu, and Yi Min Wang. Efficient multiple-click models in web search. In WSDM, pages 124–131, New York, NY, USA, 2009b. ACM Press. doi: 10.1145/1498759.1498818

[5] Fan Guo, Chao Liu, and Yi Min Wang. Efficient multiple-click models in web search. In WSDM, pages 124–131, New York, NY, USA, 2009b. ACM Press. doi: 10.1145/1498759.1498818

[6] Olivier Chapelle and Ya Zhang. A dynamic bayesian network click model for web search ranking. In WWW, pages 1–10, New York, NY, USA, 2009. ACM Press. doi: 10.1145/1526709.1526711

[7] Yuchen Zhang, Weizhu Chen, Dong Wang, and Qiang Yang. User-click modeling for understanding and predicting search-behavior. In KDD, New York, NY, USA, 2011. ACM Press