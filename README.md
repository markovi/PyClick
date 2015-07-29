PyClick - Click Models for Web Search
====

PyClick is an open-source Python library of click models for web search.
It implements most of standard click models and inference methods.

# How to Use

## Installation

```bash
cd $PROJECT_DIR
sudo python setup.py install
```


## Dependencies
For running the code:

* numpy 1.7
* enum34

For unit-testing:

* nose
* nose\_parameterized


## Running examples
Examples are located in the ```examples``` folder.
Data samples are in ```examples/data```.
Examples can be run as follows:

```
python examples/SimpleExample.py $CLICK_MODEL examples/data/YandexRelPredChallenge $SESSION_NUM
```

Here, ```$CLICK_MODEL``` is the click model to use for this example (see the list of implemented models below);
```$SESSION_NUM``` is the number of search sessions to consider.

Currently, the following click models are implemented and can be used for this example:

* GCTR (global CTR, aka, random click model)
* RCTR (rank-based CTR)
* DCTR (document-based CTR)
* PBM (position-based model)
* CM (cascade model)
* UBM (user browsing model)
* DCM (dependent click model)
* DBN (dynamic Bayesian network)
* SDBN (simplified DBN)


## Implementing a new click model
1. Inherit from ```pyclick.click_models.ClickModel```
  
        class NewClickModel(ClickModel)

2. Define the names of the model parameters:

        param_names = Enum('NCMParamNames', 'one_param another_param')

3. Choose appropriate containers for these parameters (see more on this below).
Usually, relevance-related parameters depend on a query and a document
and so can be stored in ```QueryDocumentParamContainer```.
Examination-related parameters usually depend on the document rank
and so can be stored in either ```RankParamContainer``` or ```RankSquaredParamContainer```.
Sometimes, there is a single examination parameter, stored in ```SingleParamContainer```.

4. Choose an appropriate inference method for the click model (see more on this below).
If all random variables of the model are observed, use ```MLEInference```.
Otherwise, use ```EMInference```. For other options see below.

5. Initialize the click model using the chosen parameter names, containers and inference:

        def __init__(self):
            # Specific containers are used just for the purpose of example
            self.params = {self.param_names.one_param: QueryDocumentParamContainer(),
                           self.param_names.another_param: RankSquaredParamContainer.default()}
            # MLE inference is used just for the purpose of example
            self._inference = MLEInference()
            
6. Implement model parameters.
Note that the parameter implementation usually depends on the chosen inference method,
so the same model can have different implementations of its parameters
for different inference methods.
For example, the standard DBN model uses the EM inference,
while its simplified version uses the MLE inference.
These two versions of DBN need different implementations of the DBN parameters.
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
For ready-to-use updating formulas of standard click models,
please refer to Chapter 4 of our book.
Updating formulas for new click models
can also be derived based on instructions of Chapter 4.

        Aleksandr Chuklin, Ilya Markov, Maarten de Rijke.
        Click Models for Web Search.
        Morgan & Claypool Publishers, 2015.
        
7. Implement abstract methods inherited from ```pyclick.click_models.ClickModel```.

  * ```get_session_params```: TODO
  * ```get_conditional_click_probs```: TODO
  * ```predict_click_probs```: TODO


#### Containers of click model parameters (```ParamContainer```)
TODO


#### Inference methods for click models (```Inference```)
TODO


## Acknowledgements

* The project is partially funded by the grant P2T1P2\_152269 of the Swiss National Science Foundation.
* Initially inspired by the [clickmodels](https://github.com/varepsilon/clickmodels) project.
* Contributors: Ilya Markov, Aleksandr Chuklin, Artem Grotov, Luka Stout, Finde Xumara, Bart Vredebregt, Nick de Wolf.