from cgitb import text
from re import M
from flask import Flask,render_template, request, session, url_for
from flask_session import Session
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from TTS.utils.manage import ModelManager
from TTS.utils.synthesizer import Synthesizer
import numpy as np
from logmmse import logmmse
from transformers import AutoProcessor, AutoModelForCTC
import librosa
import torch
from scipy.io.wavfile import write


import os

SESSION_TYPE = "memcache"
CHINESE_TO_ENGLISH = "c2e"
ENGLISH_TO_CHINESE = "e2c"
UPLOAD_FOLDER = os.getcwd() + "/static/uploads"

app = Flask(__name__)
app.secret_key = "f2ca1bb6c7e907d06dafe4687e579fce76b37e4e93b7605022da52e6ccc26fd2"

tokenizer_eng2chi = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-zh")
model_eng2chi = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-zh")

tokenizer_chi2eng = AutoTokenizer.from_pretrained("opus-mt-zh-en")
model_chi2eng = AutoModelForSeq2SeqLM.from_pretrained("opus-mt-zh-en")


manager = ModelManager()
model_path, config_path, model_item = manager.download_model("tts_models/zh-CN/baker/tacotron2-DDC-GST")
synthesizer_tts_chi = Synthesizer(
    model_path, config_path, None, None, None,
)

manager = ModelManager()
model_path, config_path, model_item = manager.download_model("tts_models/en/ek1/tacotron2")
synthesizer_tts_eng = Synthesizer(
    model_path, config_path, None, None, None,
)



def uc(lang):
    if(lang == CHINESE_TO_ENGLISH):
        return "Chinese"
    else:
        return "English"


def handleTextBased(type=CHINESE_TO_ENGLISH,text=""):

    if os.path.exists("audio.wav"):
        os.remove("audio.wav")
    else:
        print("The file does not exist")
    if(text==""):
        raise Exception("Text Not found")
    if(type==CHINESE_TO_ENGLISH):
        

        #chinese.py
        inputs = tokenizer_chi2eng(text,return_tensors="pt")

        outputs = model_chi2eng.generate(inputs["input_ids"], max_length=40, num_beams=4, early_stopping=True)

        output_txt = tokenizer_chi2eng.decode(outputs[0], skip_special_tokens=True)

        #tts english
        wavs = synthesizer_tts_eng.tts(output_txt)

        enhanced = logmmse(np.array(wavs, dtype=np.float32), synthesizer_tts_eng.output_sample_rate, output_file=None, initial_noise=1, window_size=160, noise_threshold=0.15)
        #write('audio.wav',synthesizer_tts_eng.output_sample_rate, data = enhanced)

    
        # handle that
        pass
    elif(type==ENGLISH_TO_CHINESE):
        # handle that
            #english

        inputs = tokenizer_eng2chi(text,return_tensors="pt")

        outputs = model_eng2chi.generate(inputs["input_ids"], max_length=40, num_beams=4, early_stopping=True)

        output_txt = tokenizer_eng2chi.decode(outputs[0], skip_special_tokens=True)


            #tts chinese
        wavs = synthesizer_tts_chi.tts(output_txt)

        enhanced = logmmse(np.array(wavs, dtype=np.float32), synthesizer_tts_chi.output_sample_rate, output_file=None, initial_noise=1, window_size=160, noise_threshold=0.15)
        write('audio.wav',synthesizer_tts_chi.output_sample_rate, data = enhanced)


        pass

    ###return r"/home/pacchu/Music/Dido - Thank You.mp3",str(text)
    return r"audio.wav",str(output_txt)

def handleSpeechBased(type=CHINESE_TO_ENGLISH,file=None):
    if(file==None):
        raise FileNotFoundError()
    if(type==CHINESE_TO_ENGLISH):

        file.save('temp.wav')
            #sstchinese

        processor_sst_chi = AutoProcessor.from_pretrained("wav2vec2-large-xlsr-53-chinese-zh-cn-gpt")
        model_sst_chi = AutoModelForCTC.from_pretrained("wav2vec2-large-xlsr-53-chinese-zh-cn-gpt")

        audio, sampling_rate = librosa.load("temp.wav")

        input_values = processor_sst_chi(audio, return_tensors = 'pt').input_values

        logits = model_sst_chi(input_values).logits

        predicted_ids = torch.argmax(logits, dim =-1)

        transcriptions = processor_sst_chi.decode(predicted_ids[0])

            #chinses.py
        inputs = tokenizer_chi2eng(transcriptions,return_tensors="pt")

        outputs = model_chi2eng.generate(inputs["input_ids"], max_length=40, num_beams=4, early_stopping=True)

        output_txt = tokenizer_chi2eng.decode(outputs[0], skip_special_tokens=True)

        
            #tts english
        wavs = synthesizer_tts_eng.tts(output_txt)

        enhanced = logmmse(np.array(wavs, dtype=np.float32), synthesizer_tts_eng.output_sample_rate, output_file=None, initial_noise=1, window_size=160, noise_threshold=0.15)
        write('audio.wav',synthesizer_tts_eng.output_sample_rate, data = enhanced)


        # handle that
        pass
    elif(type==ENGLISH_TO_CHINESE):
        
        file.save('temp.wav')

        #sstenglish
        processor_sst_eng = AutoProcessor.from_pretrained("speech-to-text")
        model_sst_eng = AutoModelForCTC.from_pretrained("speech-to-text")
        audio, sampling_rate = librosa.load("temp.wav")

        input_values = processor_sst_eng(audio, return_tensors = 'pt').input_values

        logits = model_sst_eng(input_values).logits

        predicted_ids = torch.argmax(logits, dim =-1)

        transcriptions = processor_sst_eng.decode(predicted_ids[0])


        #english


        inputs = tokenizer_eng2chi(text,return_tensors="pt")

        outputs = model_eng2chi.generate(inputs["input_ids"], max_length=40, num_beams=4, early_stopping=True)

        output_txt = tokenizer_eng2chi.decode(outputs[0], skip_special_tokens=True)


        #tts chinese
        wavs = synthesizer_tts_chi.tts(output_txt)

        enhanced = logmmse(np.array(wavs, dtype=np.float32), synthesizer_tts_chi.output_sample_rate, output_file=None, initial_noise=1, window_size=160, noise_threshold=0.15)
        write('audio.wav',synthesizer_tts_chi.output_sample_rate, data = enhanced)


        # handle that
        pass
    return r"audio.wav",output_txt,transcriptions

@app.route("/")
def main(): 
    return render_template("main.html",MAIN={
        "title":"ChineseTextEngine",
        "heading":"Translator Engine"
    })


@app.route("/tts",methods=["POST"])
def texttospeechHandler():
    if request.method == "POST":
        oglang = session.get("lang")
        text = ''
        file = request.files['audiofile']
        if(file.filename == ''):
            audiopath,text = handleTextBased(type=str(oglang),text=str(request.form["textdata"]))
            return render_template("output.html",MAIN={
                                                        "title":"ChineseTextEngine",
                                                        "heading":"Output",
                                                        "original":request.form["textdata"],
                                                        "translated":text,
                                                        "audiogen":audiopath
                                                        })
        else:
            
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
