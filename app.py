import boto3
from flask import Flask,render_template,request
import pickle
import numpy as np
from decimal import Decimal
model = pickle.load(open('model3.pkl','rb'))
app = Flask(__name__)
dynamodb = boto3.resource('dynamodb',region_name='us-east-1')
table = dynamodb.Table('employee_pred')




@app.route('/')
def hello_world():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict_placement():
    avg_training_score = Decimal(request.form.get('avg_training_score'))
    length_of_service = int(request.form.get('length_of_service'))
    age = int(request.form.get('age'))
    previous_year_rating = Decimal(request.form.get('previous_year_rating'))

    table.put_item(
        Item={
            'avg_training_score': avg_training_score,
            'previous_year_rating': previous_year_rating,
            'age': age,
            'length_of_service': length_of_service
        }
    )

    #prediction
    result = model.predict(np.array([avg_training_score,length_of_service,age,previous_year_rating]).reshape(1,4))

    if result[0] == 1:
        result = 'promoted'
    else:
        result = 'not promoted'
    return render_template('index.html',result=result)

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=8080)
