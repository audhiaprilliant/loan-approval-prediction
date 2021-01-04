# Import module
from flask import Flask, render_template, url_for, request
import pandas as pd
import numpy as np
import joblib
import pickle
import json
import os

# Current directory
current_dir = os.path.dirname(__file__)

# Function
def ValuePredictor(to_predict_list):
    loaded_model = joblib.load(open(os.path.join(current_dir,'model/xgboostModel.pkl'),'rb'))
    result = loaded_model.predict(to_predict_list)
    return result[0]

app = Flask(__name__)
app.config['DEBUG'] = True

# Index page
@app.route('/')
@app.route('/index')
def home():
    return render_template('home.html')

# Result of prediction
@app.route('/result', methods = ['POST'])
def predict():
	if request.method == 'POST':
	    # Get the data from the form
	    first_name = request.form['First']
	    gender_male = request.form['Gender_Male']
	    education_not_graduate = request.form['Education_Not Graduate']
	    self_employed_yes = request.form['Self_Employed_Yes']
	    married_yes = request.form['Married_Yes']
	    dependents = request.form['Dependents']
	    applicant_income = request.form['ApplicantIncome']
	    co_applicant_income = request.form['CoapplicantIncome']
	    loan_amount = request.form['LoanAmount']
	    loan_amount_term = request.form['Loan_Amount_Term']
	    credit_history = request.form['Credit_History_1.0']
	    property_area = request.form['Property_Area']
	    
	    # Load JSON file of columns name
	    with open(os.path.join(current_dir,'bin/columns.json'), 'r') as f:
	    	cols =  json.loads(f.read())
	    cols = cols['data_columns']

	    # Parse the categorical columns
	    new_vector = np.zeros(len(cols))
	    try:
	        new_vector[cols.index('Dependents_' + str(dependents))] = 1
	    except:
	        pass
	    try:
	    	new_vector[cols.index('Property_Area_' + str(property_area))] = 1
	    except:
	    	pass
	    # Parse the numerical columns
	    new_vector[0] = applicant_income
	    new_vector[1] = co_applicant_income
	    new_vector[2] = loan_amount
	    new_vector[3] = loan_amount_term
	    new_vector[4] = gender_male
	    new_vector[5] = married_yes
	    new_vector[9] = education_not_graduate
	    new_vector[10] = self_employed_yes
	    new_vector[11] = credit_history

	    # Convert into dataframe
	    df = pd.DataFrame(new_vector).transpose().astype(int)

	    # Change columns name
	    df.columns = cols
	   
	    # Create prediction
	    result = ValuePredictor(df)	   
	    if int(result) == 1:
	        prediction = 'Dear Mr/Mrs/Ms {first}, your loan is approved!'.format(first = first_name)
	    else:
	        prediction = 'Sorry Mr/Mrs/Ms {first}, your loan is rejected!'.format(first = first_name)
	    return render_template('result.html', prediction = prediction)

if __name__ == '__main__':
    app.run()
