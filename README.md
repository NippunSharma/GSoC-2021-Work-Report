<p align="center">
  <img width="556" height="112" src="https://github.com/NippunSharma/GSoC-2021-Work-Report/blob/main/logo.png">
</p>

#### Organisation: [mlpack](https://github.com/mlpack)

#### Project: Revamp mlpack bindings

#### Mentor: [Ryan Curtin](https://github.com/rcurtin), [James Balamuta](https://github.com/coatless) and [Yashwant Singh](https://github.com/yashwants19)

## Abstract
My proposal `Revamp mlpack bindings` was selected for Google Summer of Code 2021 under `mlpack`. `mlpack` is an intuitive, fast and flexible `C++` machine
learning library with bindings to other languages. API design is one of the most important part of any library. A complex API design will make it difficult for the user to write the program and the user will need to look at the documentation constantly, which can cause a lot of time wastage for the user. On the other hand, a very simple API design can limit the usage of the library and the user will not be able to use all the features provided. 
So, we should aim to create an intuitive API, that is not very complex but still captures most of the features required to build on.

## Goal
The project proposes to revamp the existing mlpack bindings in the Python programming language. `mlpack` is one of the most flexible machine learning libraries when accessed through C++, but when it comes to the interface in different languages mlpack still has some ground to cover. This is not intentional rather it shows how much mlpack has grown over the years. Initially, mlpack only had a command-line interface for which it was obvious to provide as much functionality as possible inside a single function. Now that mlpack not only provides a command-line interface but also an interface in Python, Julia, GO, and R, it has become a necessity to remove the single function interface and use a more modern interface with which the user is more familiar.

## Results
We were successfully able to open PR [#3030](https://github.com/mlpack/mlpack/pull/3030), that aims to merge the complete framework for this refactoring as well as changes the python bindings
for the  `adaboost` and `linear_regression` programs. The PR is almost complete and will be merged very soon.
Apart from this, we also faced an issue with the threadsafety of the `IO` class. To solve this Ryan managed to open [#2995](https://github.com/mlpack/mlpack/pull/2995) that resolved the issue,
I helped in the threadsafety issue by refactoring the Python codebase in [this](https://github.com/rcurtin/mlpack/pull/2) PR.

## Pull Requests created

### 1. [Threadsafe io Python](https://github.com/rcurtin/mlpack/pull/2)

#### Aim
1. Refactor Python binding codebase to accompany threadsafe changes.

#### Status: Merged.

### 2. [REVAMP MLPACK BINDINGS](https://github.com/mlpack/mlpack/pull/3030)

#### Aim
1. Add framework for revamping mlpack bindings in python.
2. Refactor adaboost and linear regression bindings for python.
3. Refactor markdown bindings for corresponding documentation changes.
4. Refactor tests for adaboost and linear_regression programs.
5. Make python wrappers scikit-learn compatible.
6. Make sure that wrappers work even if sciki-learn is absent.

#### Status: To be merged soon.

### 3. [[DRAFT] Revamp mlpack bindings](https://github.com/mlpack/mlpack/pull/2961)

#### Aim
1. Experiment with different ideas for revamping.
2. Experiment with CMake code.

#### Status: Closed.

## Issues opened

### 1. [Default values of Adaboost program are not good](https://github.com/mlpack/mlpack/issues/3010)

#### Aim
1. Improve the default values of the adaboost program.

#### Status: Open.


## Main Obstacles
During the course of this project, we faced a lot of obstacles. Major ones are:

1. The first obstacle faced was that the current IO structure and the way bindings are generated did
   not allow to make bindings for different methods of the same program to be defined in the same file.
2. The second major obstacle was when we decided that we wished to make the mlpack models compatible
   with scikit-learn utilities.
3. I also faced some issues while setting up the mlpack documentation website locally for changes
   in the documentation. Thankfully, Ryan helped me here and set up a dev website on his website
   so that I can make the changes.

## Blog
You can follow my progress on my blog [here](https://nippunsharma.github.io).

## Result of the project (PR to be merged soon into mlpack master branch).

#### Before refactoring.

```python
from mlpack import Adaboost
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

X, y = load_digits(n_class=10, return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, test_size=.25, random_state=0)

# training mlpack model, whose hyperparameters cannot be tuned.
# results, as expected are not good.
preds = adaboost(training=X_train, labels=y_train, test=X_test)
f1_score(y_test, preds, average="weighted")
# output:
# 0.0585 (not good)
```

#### After refactoring.

```python
from skopt import BayesSearchCV
from mlpack import Adaboost
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import numpy as np

X, y = load_digits(n_class=10, return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, test_size=.25, random_state=0)

# After the refactoring, mlpack models are compatible with
# skopt as well as scikit-learn utils.
# Here, the utliity used is BayesSearchCV() that is a hyper-
# parameter tuner.
opt = BayesSearchCV(
    Adaboost(),
    {
        'iterations': (100, 500),
        'weak_learner': ['perceptron', 'decision_stump']
    },
    n_iter=10,
    cv=3,
    scoring="f1_weighted",
    random_state=1,
)

opt.fit(X_train, y_train)

# as expected, the result becomes much better.
print("val. score: %s" % opt.best_score_)
print("test score: %s" % opt.score(X_test, y_test))
# output: 
# val. score: 0.9554649920291841
# test score: 0.9558136593062766 (much better)

opt.best_params_
# output:
# OrderedDict([('iterations', 226), ('weak_learner', 'perceptron')])
```

## Scope and Future Work
I will continue contributing to mlpack. Firstly, I will finish my open PR and get it merged as soon
as possible. Then I can help with revamping bindings in other languages as well.

## Acknowledgement
I am grateful to my mentors: Ryan Curtin, James Balamuta and Yashwant Singh for selecting my proposal and constantly
supporting and motivating me to make my project successful.

Apart from my mentors, I am also thankful to the complete mlpack community that is so friendly towards beginners. `mlpack` definitely has
one of the best open source communities. This is my first open source project that I contributed to and I will continue contributing
in the future.

Finally, I would like to thank Google for this great opportunity.
