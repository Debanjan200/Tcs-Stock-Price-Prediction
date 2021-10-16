from flask import Flask,render_template,request
import Tcs_Stock_prediction as p

app=Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")


@app.route('/predict',methods=["POST"])
def predict():

    if request.method=="POST":
        open=request.form["open"]
        high=request.form["high"]
        low=request.form["low"]
        close=request.form["close"]
        volume=request.form["volume"]

        test=[[open,high,low,close,volume]]
        test=p.sc_x.transform(test)
        pred=p.sc_y.inverse_transform(p.lin_reg.predict(test))

        return render_template("index.html",Prediction=pred[0][0],accuracy=p.accuracy)


if __name__=="__main__":
    app.run(debug=True)