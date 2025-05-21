# Reconhecimento-de-voz-offline
Por que É Tão Difícil Fazer Computadores Entenderem Sua Voz?

PLN: Uma Montanha-Russa de Desafios
Fazer um computador entender o que você diz parece simples hoje, mas já foi uma missão quase impossível! O Processamento de Linguagem Natural (PLN), que ensina máquinas a interpretar a linguagem humana, enfrentou décadas de obstáculos. Imagine tentar decifrar uma conversa cheia de sotaques, gírias, pausas e ruídos de fundo – para um computador, isso era como resolver um quebra-cabeça sem as peças certas. Vamos mergulhar nos desafios históricos e na revolução que tornou o PLN acessível, com um aplicativo em Python para te desafiar: você consegue criar algo tão legal quanto isso?

Por que É Tão Difícil Entender a Voz Humana?
A voz humana é uma bagunça maravilhosa. Cada pessoa fala de um jeito único, com variações de tom, ritmo, sotaque e até erros. Para um computador, transformar esse caos em texto é um processo complexo que envolve vários passos técnicos:

Capturar o som: A voz começa como ondas sonoras, vibrações no ar que precisam ser convertidas em sinais elétricos. Um microfone faz isso transformando as vibrações em variações de voltagem, criando um sinal analógico.
Digitalizar o sinal: Esse sinal analógico é convertido em números (sinal digital) por um conversor analógico-digital (ADC). O áudio é "amostrado" milhares de vezes por segundo (geralmente 16.000 vezes, ou 16 kHz, para fala) e cada amostra é representada por números (ex.: 16 bits por amostra).
Processar o sinal: O sinal digital é dividido em pequenos trechos (frames) de 10-30 milissegundos. Esses trechos são analisados para extrair características acústicas, como frequências (usando técnicas como a Transformada Rápida de Fourier, ou FFT) que representam sons como vogais ou consoantes.
Reconhecer padrões: Aqui entra o PLN. O computador compara essas características com um modelo treinado para identificar fonemas (os menores sons da fala, como "p" ou "a"). Depois, junta os fonemas em palavras e frases, considerando o contexto e a gramática.
Lidar com a variabilidade: A fala humana varia muito – sotaques, ruídos de fundo, emoções, velocidade. Um modelo precisa ser treinado com milhares de horas de áudio para "aprender" essas diferenças, o que exige dados e poder computacional enormes.
No passado recente (décadas de 80 a 2000), esse processo era um pesadelo:

Conversão de áudio: Microfones e ADCs eram caros e de baixa qualidade, resultando em sinais ruidosos. Muitas vezes, o áudio era capturado em 8 kHz (metade da qualidade atual), perdendo detalhes importantes.
Modelos rudimentares: Sistemas como o Hidden Markov Model (HMM) eram comuns, mas só funcionavam bem com vocabulários pequenos e falantes específicos. Eles exigiam que o usuário "treinasse" o sistema falando frases predefinidas.
Falta de dados: Coletar horas de áudio anotado era caro e demorado. Sem grandes bancos de dados, os modelos eram limitados a contextos específicos, como comandos curtos ("ligar", "desligar").
Computação limitada: Computadores da época mal conseguiam processar áudio em tempo real. Um sistema de reconhecimento de fala podia levar minutos para transcrever uma frase!
Ferramentas fechadas: Empresas como IBM e AT&T dominavam o PLN, com sistemas proprietários caros, deixando pouco espaço para pesquisadores independentes.
Era como tentar ensinar um robô a entender poesia com uma calculadora de bolso. Mas tudo mudou com a revolução tecnológica e a colaboração open-source.

A Revolução do PLN: Tecnologia Moderna e Open-Source
Nos últimos 15-20 anos, o PLN deu um salto gigante, graças a:

Poder computacional: Hoje, até um laptop básico tem mais poder que os supercomputadores dos anos 90. GPUs e CPUs modernas processam áudio e modelos complexos em tempo real.
Bancos de dados abertos: Comunidades open-source, como o projeto Common Voice, disponibilizam milhares de horas de áudio em vários idiomas, incluindo português.
Modelos avançados: Técnicas como redes neurais profundas e transformers (usados em modelos como BERT e GPT) permitem reconhecer fala com precisão, mesmo com sotaques e ruídos.
Ferramentas open-source: Projetos como Vosk, Kaldi e Hugging Face oferecem modelos e bibliotecas gratuitos. Qualquer pessoa pode baixar um modelo de reconhecimento de fala e usá-lo sem pagar nada.
O resultado? O que antes era um privilégio de grandes empresas agora está nas mãos de qualquer curioso com um computador. Para provar, vou te mostrar um aplicativo em Python que usa o Vosk para transformar sua voz em texto – e te desafio a testá-lo e criar algo ainda mais incrível!

O Aplicativo: Sua Voz Virando Texto
Este app é um exemplo prático de PLN. Ele:

Transcreve fala em tempo real pelo microfone ou a partir de arquivos WAV.
Mostra o volume do áudio com uma barra de progresso (é bem legal ver ela pulsar!).
Usa o Vosk, que funciona offline, garantindo privacidade.
Tem uma interface simples feita com Python.
Antes de Começar
Você vai precisar de:

Python 3.x.
Bibliotecas: tkinter (incluso no Python), pyaudio, vosk e numpy.
O modelo Vosk para português (vosk-model-small-pt-0.3), disponível em https://alphacephei.com/vosk/models.
Instale as dependências com:
![Uploading image.png…]()


```
pip install pyaudio vosk numpy
```

O Código: Seu Transcritor de Voz
Aqui está o código, com comentários simples para você entender e personalizar:

```
import tkinter as tk
from tkinter import filedialog, ttk
import pyaudio
import wave
import json
from vosk import Model, KaldiRecognizer
import threading
import queue
import os
import struct
import numpy as np

class SpeechRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Vosk Speech Recognition")
        self.root.geometry("600x400")
        
        # Variáveis para controlar o app
        self.is_recognizing = False
        self.audio_queue = queue.Queue()
        self.model = None
        self.recognizer = None
        self.stream = None
        self.audio_thread = None
        
        # Monta a interface gráfica
        self.setup_ui()
        
        # Carrega o modelo Vosk
        self.load_model()
        
    def setup_ui(self):
        # Cria a tela principal
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Escolha entre microfone ou arquivo WAV
        ttk.Label(main_frame, text="Fonte de Áudio:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.audio_source = ttk.Combobox(main_frame, values=["Microfone", "Arquivo WAV"], state="readonly")
        self.audio_source.grid(row=0, column=1, sticky=tk.W, pady=5)
        self.audio_source.current(0)
        
        # Botão para selecionar arquivo WAV
        self.select_file_btn = ttk.Button(main_frame, text="Selecionar Arquivo", command=self.select_wav_file)
        self.select_file_btn.grid(row=0, column=2, padx=5)
        self.select_file_btn.config(state="disabled")
        
        # Mostra o caminho do arquivo escolhido
        self.file_path_var = tk.StringVar()
        ttk.Label(main_frame, textvariable=self.file_path_var).grid(row=1, column=0, columnspan=3, sticky=tk.W, pady=5)
        
        # Área onde aparece o texto transcrito
        self.text_area = tk.Text(main_frame, height=15, width=60, wrap=tk.WORD)
        self.text_area.grid(row=2, column=0, columnspan=3, pady=10)
        
        # Barra que mostra o volume do som
        ttk.Label(main_frame, text="Nível de Áudio:").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.volume_bar = ttk.Progressbar(main_frame, length=300, mode='determinate', maximum=100)
        self.volume_bar.grid(row=3, column=1, columnspan=2, pady=5, sticky=tk.W)
        
        # Botões para iniciar e parar
        self.start_btn = ttk.Button(main_frame, text="Iniciar Reconhecimento", command=self.start_recognition)
        self.start_btn.grid(row=4, column=0, pady=10)
        self.stop_btn = ttk.Button(main_frame, text="Parar Reconhecimento", command=self.stop_recognition, state="disabled")
        self.stop_btn.grid(row=4, column=1, pady=10)
        
        # Habilita/desabilita botão de arquivo
        self.audio_source.bind("<<ComboboxSelected>>", self.update_file_button_state)
    
    def load_model(self):
        # Carrega o modelo Vosk para português
        model_path = "vosk-model-small-pt-0.3"
        if not os.path.exists(model_path):
            self.text_area.insert(tk.END, "Erro: Modelo Vosk não encontrado. Baixe em https://alphacephei.com/vosk/models\n")
            return
        self.model = Model(model_path)
        self.recognizer = KaldiRecognizer(self.model, 16000)
    
    def update_file_button_state(self, event=None):
        # Libera o botão de arquivo se escolher WAV
        self.select_file_btn.config(state="normal" if self.audio_source.get() == "Arquivo WAV" else "disabled")
        self.file_path_var.set("")
    
    def select_wav_file(self):
        # Abre janela para escolher arquivo WAV
        file_path = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
        if file_path:
            self.file_path_var.set(file_path)
    
    def calculate_volume(self, data):
        try:
            # Calcula o volume do som para a barra
            fmt = f"{len(data)//2}h"
            samples = struct.unpack(fmt, data)
            rms = np.sqrt(np.mean(np.square(samples)))
            max_rms = 32768  # Máximo para áudio 16-bit
            volume = min(100, (rms / max_rms) * 100 * 2)
            return volume
        except:
            return 0
    
    def start_recognition(self):
        # Começa a transcrição
        if not self.model:
            self.text_area.insert(tk.END, "Erro: Modelo não carregado.\n")
            return
        if self.is_recognizing:
            return
        
        self.is_recognizing = True
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.text_area.delete(1.0, tk.END)
        
        source = self.audio_source.get()
        if source == "Microfone":
            self.audio_thread = threading.Thread(target=self.process_microphone)
            self.audio_thread.daemon = True
            self.audio_thread.start()
        elif source == "Arquivo WAV" and self.file_path_var.get():
            self.audio_thread = threading.Thread(target=self.process_wav_file, args=(self.file_path_var.get(),))
            self.audio_thread.daemon = True
            self.audio_thread.start()
        else:
            self.text_area.insert(tk.END, "Erro: Selecione um arquivo WAV válido.\n")
            self.stop_recognition()
        
        # Atualiza o texto na tela
        self.root.after(100, self.update_text)
    
    def stop_recognition(self):
        # Para a transcrição
        self.is_recognizing = False
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        self.volume_bar.config(value=0)
    
    def process_microphone(self):
        # Captura áudio do microfone
        p = pyaudio.PyAudio()
        self.stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=4000)
        while self.is_recognizing:
            data = self.stream.read(4000, exception_on_overflow=False)
            volume = self.calculate_volume(data)
            self.root.after(0, lambda: self.volume_bar.config(value=volume))
            if self.recognizer.AcceptWaveform(data):
                result = json.loads(self.recognizer.Result())
                if result.get("text"):
                    self.audio_queue.put(result["text"])
        self.audio_queue.put(json.loads(self.recognizer.FinalResult()).get("text", ""))
        self.stream.stop_stream()
        self.stream.close()
        p.terminate()
        self.root.after(0, lambda: self.volume_bar.config(value=0))
    
    def process_wav_file(self, file_path):
        # Processa arquivo WAV
        try:
            with wave.open(file_path, "rb") as wf:
                if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() != 16000:
                    self.audio_queue.put("Erro: O arquivo WAV deve ser mono, 16-bit, 16kHz.\n")
                    return
                while self.is_recognizing:
                    data = wf.readframes(4000)
                    if len(data) == 0:
                        break
                    volume = self.calculate_volume(data)
                    self.root.after(0, lambda: self.volume_bar.config(value=volume))
                    if self.recognizer.AcceptWaveform(data):
                        result = json.loads(self.recognizer.Result())
                        if result.get("text"):
                            self.audio_queue.put(result["text"])
                self.audio_queue.put(json.loads(self.recognizer.FinalResult()).get("text", ""))
        except Exception as e:
            self.audio_queue.put(f"Erro ao processar o arquivo: {str(e)}\n")
        self.root.after(0, lambda: self.volume_bar.config(value=0))
    
    def update_text(self):
        # Mostra o texto transcrito na tela
        try:
            while not self.audio_queue.empty():
                text = self.audio_queue.get_nowait()
                if text:
                    self.text_area.insert(tk.END, text + "\n")
                    self.text_area.see(tk.END)
        except queue.Empty:
            pass
        if self.is_recognizing:
            self.root.after(100, self.update_text)
        else:
            self.recognizer = KaldiRecognizer(self.model, 16000)  # Reseta o reconhecedor
    
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    root = tk.Tk()
    app = SpeechRecognitionApp(root)
    app.run()
```
Como o App Funciona?
Este aplicativo é como um tradutor de voz para texto. Ele:

Usa tkinter para criar uma janela com botões e uma área de texto.
Usa Vosk para transformar áudio em texto, processando pequenos trechos de som (frames) e analisando-os com um modelo treinado.
Usa PyAudio para capturar o áudio do microfone, convertendo ondas sonoras em sinais digitais (16 bits, 16 kHz, mono).
Mostra uma barra de progresso que pulsa com o volume, calculado como a raiz quadrada da média dos quadrados das amostras (RMS).
No passado, um app assim exigiria um supercomputador e meses de trabalho. Hoje, graças ao Vosk e ao Python, você pode rodá-lo no seu laptop!

Desafio: Você Pode Ir Além?
Aqui vai a provocação: e se você pegasse esse código e fizesse algo que até os pesquisadores dos anos 90 invejariam? Tente:

Adicionar um botão para salvar o texto transcrito em um arquivo.
Testar modelos Vosk em outros idiomas (inglês, espanhol, etc.).
Criar um sistema que detecta palavras-chave e dispara ações, como acender uma luz!
Você topa o desafio? Baixe o modelo Vosk, instale as bibliotecas e teste o app. Consegue transcrever sua voz ou melhorar o código? Conta nos comentários o que conseguiu ou que ideias malucas teve!

O Poder do Open-Source
O Vosk é um exemplo brilhante de como a comunidade open-source mudou o PLN. Projetos como Vosk, Kaldi e Hugging Face oferecem modelos e ferramentas gratuitas, construídas por milhares de colaboradores. Eles transformaram o PLN de um campo restrito a laboratórios caros em algo que qualquer pessoa com um computador pode explorar. É como se o mundo todo estivesse construindo o futuro da IA juntos!

Vamos Criar o Futuro do PLN?
Este app é só o começo. Com o PLN mais acessível do que nunca, você pode fazer parte dessa revolução. Rode o código, fale no microfone e veja sua voz virar texto. O que você vai criar com essa tecnologia? Deixe nos comentários suas experiências, dúvidas ou ideias para levar o PLN ao próximo nível. Para mais sobre o Vosk, visite https://alphacephei.com/vosk/. Vamos transformar sons em ideias juntos!
