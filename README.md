![GitHub License](https://img.shields.io/github/license/views-platform/views-stepshifter)
![GitHub branch check runs](https://img.shields.io/github/check-runs/views-platform/views-stepshifter/main)
![GitHub Issues or Pull Requests](https://img.shields.io/github/issues/views-platform/views-stepshifter)
![GitHub Release](https://img.shields.io/github/v/release/views-platform/views-stepshifter)


<div style="width: 100%; max-width: 1500px; height: 400px; overflow: hidden; position: relative;">
  <img src="https://github.com/user-attachments/assets/1ec9e217-508d-4b10-a41a-08dface269c7" alt="VIEWS Twitter Header" style="position: absolute; top: -50px; width: 100%; height: auto;">
</div>


# **Stepshifter**: Time-series Forecasting Model 🔮  

> **Part of the [VIEWS Platform](https://github.com/views-platform) ecosystem for large-scale conflict forecasting.**

---

## 📚 Table of Contents  

1. [Overview](#overview)  
2. [Role in the VIEWS Pipeline](#role-in-the-views-pipeline)  
3. [Features](#features)  
4. [Installation](#installation)  
5. [Usage](#usage)  
6. [Architecture](#architecture)  
7. [Project Structure](#project-structure)  
8. [Contributing](#contributing)  
9. [License](#license)  
10. [Acknowledgements](#acknowledgements)  

---

## 🧠 Overview  
**Stepshifter** is a machine learning model designed for **time-series forecasting** using [Darts](https://unit8co.github.io/darts/). 
It solves **[regression and classification]** tasks. 

**Key Capabilities**:
- **Probabilistic Outputs**: Binary outputs and point predictions.       
- **Learning Approach**:    
1. ```LinearRegressionModel```, 
2. ```RandomForest```, 
3. ```LightGBMModel```, 
4. ```XGBModel```,
5. ```HurdleModel```.  
- **Integration-Ready**: Built to seamlessly integrate into the larger **VIEWS Pipeline**. 


---

## 🌍 Role in the VIEWS Pipeline  

**Stepshifter** serves as part of the **Violence & Impacts Early Warning System (VIEWS)** pipeline. It complements the following repositories:  

- **[views-pipeline-core](https://github.com/views-platform/views-pipeline-core):** For data ingestion and preprocessing.  
- **[views-models](https://github.com/views-platform/views-models):** Handles training, testing, and deployment.  
- **[views-evaluation](https://github.com/views-platform/views-evaluation):** Evaluates and calibrates model outputs.  
- **[docs](https://github.com/views-platform/docs):** Organization/pipeline level documentation.


### Integration Workflow  

**Stepshifter** fits into the pipeline as follows:  
1. **Data Input:** Processes preprocessed data from [**views-pipeline-core**](https://github.com/views-platform/views-pipeline-core).  
2. **Model Execution:** This modeling approach involves shifting all independent variables in time, in order to train models that can predict future values of the dependent variable. 
3. **Post-Processing:** Outputs are sent to [**views-evaluation**](https://github.com/views-platform/views-evaluation) for further analysis.

---

## ✨ Features  

- **Darts models**: Stepshifter model class supports multiple Darts forecasting models, including ```LinearRegressionModel```, ```RandomForest```, ```LightGBMModel```, and ```XGBModel```. 
- **Automated Data Cleanup**: Stepshifter model class automatically processes missing data and ensures consistent multi-index formatting for time-series data.
- **Hurdle model**: Hurdle model class inherits from StepshifterModel.
 A hurdle model consists of two stages:
    1. Binary stage: Predicts whether the target variable is 0 or > 0.
    2. Positive stage: Predicts the value of the target variable when it is > 0.



---

## ⚙️ Installation  

### Prerequisites  

- Python >= 3.11 
- Access to **views-pipeline-core**.


### Steps  

See the organization/pipeline level [docs](https://github.com/views-platform/docs)  


---

## 🚀 Usage  

### 1. Run Training Locally  

See the organization/pipeline level [docs](https://github.com/views-platform/docs)  

### 2. Use in the VIEWS Pipeline  

Stepshifter integrates seamlessly with the VIEWS pipeline. After processing, outputs can be passed to **views-evaluation** for further calibration or ensembling.  

---

## 🏗 Architecture  

### 1. Stepshifter Model

This modeling approach involves shifting all independent variables in time, in order to train models that can predict future values of the dependent variable. More details can be found in [Appendix A of Hegre et al. (2020)](https://viewsforecasting.org/wp-content/uploads/2020/09/AppendixA.pdf).

### 2. Hurdle Model


This approach differs from a traditional implementation in three aspects:
1. In the first stage, since Darts doesn't support classification models, a regression model is used instead. These estimates are not strictly bounded between 0 and 1, but this is acceptable for the purpose of this step.
2. To determine whether an observation is classified as "positive," we apply a threshold. The default threshold is 1, meaning that predictions above this value 
are considered positive outcomes. It is not set as 0 because most predictions won't be exactly 0. This threshold can be adjusted as a tunable hyperparameter to better suit specific requirements.
3.  In the second stage, a regression model is used to predict for the selected time series. Since Darts time series require a continuous timestamp, we can't get rid of those timestamps with negative prediction produced in the first stage like a traditional implementation. Instead we include the entire time series for countries or PRIO grids where the first stage yielded at least one positive prediction.


## 🚦 Workflow  


1. **Input:** VIEWS historical conflict data.  
2. **Processing:** Converting to Darts time series data.  
3. **Prediction:** Regression predictions.  

Refer to the **[Appendix A of Hegre et al. (2020)](https://viewsforecasting.org/wp-content/uploads/2020/09/AppendixA.pdf)** for an in-depth explanation.

For more detailed information about the VIEWS Stepshifter models themselves, refer to the [VIEWS models catalog](https://github.com/views-platform/views-models/). 

---

## 🗂 Project Structure  

```plaintext
views-stepshifter/
├── README.md          # Documentation
├── tests              # Unit and integration tests
├──  views-stepshifter # Main source code
│   ├── manager        # Management of stepshifter model lifecycle
│   ├── models         # Model algorithms
│   ├── src            # Folder template
│   ├── __init__.py    # Package initialization
├── .gitignore         # Git ignore rules
├── pyproject.toml     # Poetry project file
├── poetry.lock        # Dependency lock file
```  

---

## 🤝 Contributing  

We welcome contributions to this project! Please follow the guidelines in the [VIEWS Documentation](https://github.com/views-platform/docs).  


---

## 📜 License  

This project is licensed under the [LICENSE](/LICENSE) file.  

---

## 💬 Acknowledgements  


<p align="center">
  <img src="https://raw.githubusercontent.com/views-platform/docs/main/images/views_funders.png" alt="Views Funders" width="80%">
</p>



Special thanks to the **VIEWS MD&D Team** for their collaboration and support.  

