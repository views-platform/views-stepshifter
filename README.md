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

## Example
Please refer to [tutorial.py](https://github.com/views-platform/views-stepshifter/blob/main/tutorial.ipynb) for more details.