from flask import Flask,render_template, request, session, url_for
from flask_session import Session
import os

SESSION_TYPE = "memcache"
CHINESE_TO_ENGLISH = "c2e"
ENGLISH_TO_CHINESE = "e2c"
UPLOAD_FOLDER = os.getcwd() + "/static/uploads"

app = Flask(__name__)
app.secret_key = "f2ca1bb6c7e907d06dafe4687e579fce76b37e4e93b7605022da52e6ccc26fd2"

def uc(lang):
    if(lang == CHINESE_TO_ENGLISH):
        return "Chinese"
    else:
        return "English"


def handleTextBased(type=CHINESE_TO_ENGLISH,text=""):
    if(text==""):
        raise Exception("Text Not found")
    if(type==CHINESE_TO_ENGLISH):
        # handle that
        pass
    elif(type==ENGLISH_TO_CHINESE):
        # handle that
        pass
    return "some",str(text)

def handleSpeechBased(type=CHINESE_TO_ENGLISH,file=None):
    if(file==None):
        raise FileNotFoundError()
    if(type==CHINESE_TO_ENGLISH):
        # handle that
        pass
    elif(type==ENGLISH_TO_CHINESE):
        # handle that
        pass
    return r"/home/pacchu/Music/Dido - Thank You.mp3","TranscribedText",str(text)

@app.route("/")
def main(): 
    return render_template("main.html",MAIN={
        "title":"ChineseTextEngine",
        "heading":"Translator Engine"
    })


@app.route("/tts",methods=["POST"])
def texttospeechHandler():
    print(request.form)
    if request.method == "POST":
        oglang = session.get("lang")
        print(request.form)
        text = ''
        file = None

        file = request.files['audiofile']
        if(len(file.filename) < 1):
            print(request.form)
            audiopath,text = handleTextBased(type=str(oglang),text=str(request.form["txtdta"]))
            return render_template("output.html",MAIN={
                                                        "title":"ChineseTextEngine",
                                                        "heading":"Output",
                                                        "original":str(request.form["txtdta"]),
                                                        "translated":text,
                                                        "audiogen":audiopath
                                                        })
        else:
            print(f"Got file : {file.filename}")
            audiopath,transcribe,text = handleSpeechBased(type=str(oglang),file=file)
            return render_template("output.html",MAIN={
                                                        "title":"ChineseTextEngine",
                                                        "heading":"Output",
                                                        "original":transcribe,
                                                        "translated":text,
                                                        "audiogen":audiopath
                                                        })    
    else:
        return "500?"

@app.route("/trans",methods=["GET", "POST"])
def trans():
    lang = CHINESE_TO_ENGLISH if request.args.get("ltype") == "c2e" else ENGLISH_TO_CHINESE
    to_say = "输入中文文本" if lang == CHINESE_TO_ENGLISH else "Enter text in English"
    session["lang"] = lang
    # if(request.args.get("ttype")=="tts"):
    return render_template("tts.html",MAIN={
            "title":"Text To Speech",
            "heading":"Text to speech Engine",
            "oglang":to_say
        })
    # else:
    #     return render_template("stt.html",MAIN={
    #         "title":"Speech To Text",
    #         "heading":"Speech to text Engine",
    #         "oglang":uc(lang)
    #     })


if(__name__ == "__main__"):
    app.run("0.0.0.0",8000)
    app.debug= True