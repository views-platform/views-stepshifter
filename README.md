# views-stepshifter

A views stepshifter model is trained on views data that is shifted relative to the dependent variable. 

## Installation

1. **Intall** ```libomp``` for Mac user because some packages (e.g. lightgbm) requires extra library. 
```
brew install libomp
```

The following setup is often required when working with C/C++ libraries or Python packages when working with ```libomp```.
````
echo 'export LDFLAGS="-L/opt/homebrew/opt/libomp/lib"' >> ~/.zshrc                           
echo 'export CPPFLAGS="-I/opt/homebrew/opt/libomp/include"' >> ~/.zshrc
source ~/.zshrc
````

````
echo 'export DYLD_LIBRARY_PATH="/opt/homebrew/opt/libomp/lib:$DYLD_LIBRARY_PATH"' >> ~/.zshrc
source ~/.zshrc
````

2. **Install** ```views_stepshifter```
```
pip install views_stepshifter
```

## Overview
### 1. StepshifterManager
This class inherits from base class ModelManager. It includes:
* a class variable to distinguish between stepshifter models and hurdle models.
* functions to train, evaluate and predict and sweep.

### 2. StepshifterModel
This model class is designed for time-series forecasting using Darts. It
* supports multiple Darts forecasting models, including ```LinearRegressionModel```, ```RandomForest```, ```LightGBMModel```, and ```XGBModel```.
* automatically processes missing data and ensures consistent multi-index formatting for time-series data.

This modeling approach involves shifting all independent variables in time, in order to train models that can predict future values of the dependent variable. More details can be find in [Appendix A of Hegre et al. (2020)](https://viewsforecasting.org/wp-content/uploads/2020/09/AppendixA.pdf).

### 3. HurdleModel
This model class inherits from StepshifterModel. A hurdle model consists of two stages:
1. Binary stage: Predicts whether the target variable is 0 or > 0.
2. Positive stage: Predicts the value of the target variable when it is > 0.

This approach differs from a traditional implementation in three aspects:
1. In the first stage, since Darts doesn't support classification models, a regression model is used instead. These estimates are not strictly bounded between 0 and 1, but this is acceptable for the purpose of this step.
2. To determine whether an observation is classified as "positive," we apply a threshold. The default threshold is 1, meaning that predictions above this value 
are considered positive outcomes. It is not set as 0 because most predictions won't be exactly 0. This threshold can be adjusted as a tunable hyperparameter to better suit specific requirements.
3.  In the second stage, a regression model is used to predict for the selected time series. Since Darts time series require a continuous timestamp, we can't get rid of those timestamps with negative prediction produced in the first stage like a traditional implementation. Instead we include the entire time series for countries or PRIO grids where the first stage yielded at least one positive prediction.