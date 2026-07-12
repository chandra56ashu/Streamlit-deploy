# Climate Change Sentiment — Streamlit Prototype

A legacy Streamlit deployment experiment created to demonstrate how a text-classification model could be exposed through a simple web interface.

This repository is connected to the broader [Climate Change Tweet Classification](https://github.com/chandra56ashu/Sentimental-Analysis) project.

## Intended workflow

1. Load labelled climate-change tweet data
2. Split tweet messages into training and test sets
3. Build a pipeline using TF-IDF and Multinomial Naive Bayes
4. Evaluate the classifier
5. Save the trained pipeline as `model.pkl`
6. Load the model in a Streamlit application
7. Accept a user-entered message and display its predicted class

## Technology

- Python
- pandas and NumPy
- scikit-learn
- TF-IDF vectorisation
- Multinomial Naive Bayes
- Streamlit
- pickle

## Repository contents

- `Ml.py` — model-training and serialisation script
- `app.py` — Streamlit interface prototype
- `model.pkl` — saved classifier
- `Data.csv` — project data
- `requirements.txt` — original dependencies
- `Data Preprocessing/` — supporting preprocessing notebook

## Project status

This repository is retained as a **historical learning and deployment prototype**. It uses an older Streamlit and scikit-learn environment, and the current application code requires modernisation before reliable deployment.

Known areas to update include:

- Replace the local absolute dataset path in `Ml.py`
- Pass the user's entered message to the model correctly
- Return the model prediction rather than the prediction function object
- Update and deduplicate dependencies
- Add class-label mapping and input validation
- Rebuild the saved model under a current Python environment

## Recommended modernisation

Use a current virtual environment and update the implementation to:

```python
prediction = model.predict([message])[0]
```

Then map the returned class to a user-friendly stance label.

## Relationship to the main project

For the full exploratory analysis, preprocessing and modelling documentation, see:

**[Sentimental-Analysis](https://github.com/chandra56ashu/Sentimental-Analysis)**

## Author

Ashutosh Chandra
