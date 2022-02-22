"""

"""
import json
import requests

test_neg = {
    "age": 39,
    "workclass": "State-gov",
    "education": "Bachelors",
    "marital-status": "Never-married",
    "occupation": "Adm-clerical",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "Male",
    "hours-per-week": 40,
    "native-country": "United-States",
}

test_pos = {
    "age": 40,
    "workclass": "Private",
    "education": "Doctorate",
    "marital-status": "Married-civ-spouse",
    "occupation": "Prof-specialty",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "hours-per-week": 60,
    "native-country": "United-States",
}


def text_salary(json_res):
    """

    Parameters
    ----------
    json_res

    Returns
    -------

    """
    if json_res['Salary'] == '0':
        salary = "Below $50K"
    else:
        salary = "Above $50K"
    return salary


if __name__ == "__main__":
    response = requests.post(
        'https://fastapi-ml-deployment.herokuapp.com/predict',
        data=json.dumps(test_neg))
    print(response.status_code)
    print(text_salary(response.json()))
    print(response.json())
    response = requests.post(
        'https://fastapi-ml-deployment.herokuapp.com/predict',
        data=json.dumps(test_pos))
    print(response.status_code)
    print(text_salary(response.json()))
    print(response.json())
