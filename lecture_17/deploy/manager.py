from flask import Flask, render_template, request
import prediction

app = Flask("My-app")

@app.route("/")
def hello():
    return render_template("form.template", result="")

@app.route("/submit",  methods = ["post"])
def submit():
    text = request.form["text"]
    return render_template("form.template", result=prediction.predict(text))

@app.route("/submit",  methods = ["get"])
def back_login():
    return render_template("form.template")
