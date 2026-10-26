# Source code used in the paper "Loss-based multi-objective design optimization framework for hybrid bracing frames with partially self-centering capacity" published in *Earthquake Engineerng & Structural Dynamics* for Loss Prediction

Usage:
1. Make sure the uv tool is installed. If not, install it using ```pip install uv``` command. 
2. Install the required packages using ```uv venv``` and ```uv sync``` commands.
3. Run train1.py using ```uv run train1.py``` for the first stage of training.
4. Run train2.py using ```uv run train2.py``` for the second stage of training.
5. Run get_shap_figures using ```uv run get_shap_figures.py``` for generating SHAP figures