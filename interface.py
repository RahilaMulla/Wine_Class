
from flask import Flask , jsonify ,render_template , request
import pickle
import numpy as np

with open("Linear_model.pkl","rb")as f:
    ln_reg=pickle.load(f)
    
app = Flask(__name__)
##################################################################################
 ################################# Base API #####################################
##################################################################################

@app.route('/')



@app.route('/Homepage')
def Homepage():
    print('Welcome to Wine Class Model')
    return render_template('home.html')
    


##################################################################################
 ################################# Model API #####################################
##################################################################################

@app.route('/predict_class',methods=['POST'])
def get_wine_class():
       
    data = request.form
    alcohol             = eval(data['alcohol'])
    malic_acid          = eval(data['malic_acid'])
    ash                 = eval(data['ash'])
    alcalinity_of_ash   = eval(data['alcalinity_of_ash'])
    magnesium           = eval(data['magnesium'])
    total_phenols       = eval(data['total_phenols'])
    flavanoids          = eval(data['flavanoids'])
    nonflavanoid_phenol = eval(data['nonflavanoid_phenol'])
    proanthocyanins     = eval(data['proanthocyanins'])
    color_intensity     = eval(data['color_intensity'])
    hue                 = eval(data['hue'])
    diluted_wines       = eval(data['diluted_wines'])
    proline             = eval(data['proline'])
    
    test_array=np.zeros(13) 
    test_array[0]=alcohol
    test_array[1]=malic_acid
    test_array[2]=ash
    test_array[3]=alcalinity_of_ash
    test_array[4]=magnesium
    test_array[5]=total_phenols
    test_array[6]=flavanoids
    test_array[7]=nonflavanoid_phenol
    test_array[8]=proanthocyanins
    test_array[9]=color_intensity
    test_array[10]=hue
    test_array[11]=diluted_wines
    test_array[12]=proline
    print('Test Array :', test_array)
        
    classes=ln_reg.predict([test_array])
    return render_template('after.html', data=classes)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port = 8080, debug= False)
