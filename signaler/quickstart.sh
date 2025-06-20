pyenv install 3.11.6
pyenv local 3.11.6
echo "3.11.6" > .python-version  # Commit this file
python -m venv venv
source venv/bin/activate
pip install -e ".[dev,test,lint]"