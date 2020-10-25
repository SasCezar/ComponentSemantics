# ComponentSemantics


## Setup

### Conda

``` commandline
conda create --name [envname] --filename requirements.txt 
```

### Pip
1. Create a python venv with python version 3.8.3 
2. ```commandline 
   pip install -r requirements_pip.txt
   ```

## Usage
Move to the ```componentSemantics``` module, and run the following commands.
1. Create communities
    ```commandline
    python community_extraction.py
    ```
2. Create features
    ```commandline
    python feature_extraction.py
    ```
3. Run rimilarity measures
    ```commandline
    python similarity.py
    ```
   
## Extra
- Create plots as in the paper, from the ```scrips``` folder, open the ```paper-scatter-plot.r``` file, update the paths, 
and execute.

