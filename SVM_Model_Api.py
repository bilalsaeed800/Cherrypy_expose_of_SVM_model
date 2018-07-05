import cherrypy
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

cherrypy.config.update({'server.socket_host': '0.0.0.0',
                        'server.socket_port': 44123})

def CORS():
	cherrypy.response.headers["Access-Control-Allow-Origin"] = "*"

class RecGenerator(object):
    @cherrypy.expose
    def index(self):
        return "hello !"
    
    @cherrypy.expose
    def abc(self , dtaa='0.12,0.34,0.21'):
        dat1= dtaa.rsplit(",")
        dat1 = [float(i) for i in dat1]
        return str(dat1[0]+dat1[1]+dat1[2])
        
    @cherrypy.expose
    def getData(self, lstData= '0.688888889,0,0.145333333,0.279187817,0.331279621,0.032,0.020288091,0.78125,0.581818182,0.264285714'):
        lstData1 = lstData.rsplit(",")
        lstData2 = [float(i) for i in lstData1]
        df_test=np.array(lstData2)
        X_test=df_test[:10]
        
        model_pkl = open('E:\\8 semester\\FYP-II\\working\\train_model\\model_20180428.pkl', 'rb')
        model = pickle.load(model_pkl)
        
        y_pred=model.predict(X_test.reshape(1, -1))        
        return str(y_pred[0])
        
if __name__ == '__main__':
	conf = {
        '/': {
            'tools.response_headers.on': True,
            'tools.response_headers.headers': [('Content-Type', 'text/plain')],
			'tools.CORS.on': True
        }
    }
	cherrypy.tools.CORS = cherrypy.Tool('before_handler', CORS)	
	cherrypy.quickstart(RecGenerator(), '/', conf)
    
        
"""
df=pd.read_csv('E:/8 semester/FYP-II/dataset/ILDP.csv')
        df_train=df.iloc[:400,:]
        X_train=df_train.iloc[:,:-1]
        y_train=df_train.iloc[:,-1]
        df_test=df.iloc[400:,:]
        X_test=df_test.iloc[:,:-1]
        y_test=df_test.iloc[:,-1]
        model_pipeline=Pipeline([('scaler', StandardScaler()),
                         ('SVM', SVC())])
        model_pipeline.fit(X_train, y_train)
        y_pred=model_pipeline.predict(X_test)
        accuracy=accuracy_score(y_test, y_pred)
        conf_matrix=confusion_matrix(y_test, y_pred)
        
plt.style.use('ggplot') # Using ggplot for visualization
plt.title('Frequencies of Age')
plt.xlabel('Age')
plt.hist(df['Age'])
plt.show()

plt.title('Protiens vs Target')
plt.xlabel('Protiens')
plt.ylabel('Target')
plt.scatter(df['Total.Protiens'], df['data.Y'])
plt.show()
"""
