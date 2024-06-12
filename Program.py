import os
import sys
import numpy as np
import tensorflow as tf
import joblib
import pandas as pd
import tkinter as tk
from tkinter import messagebox, font, ttk
from fontTools.ttLib import TTFont
from tempfile import TemporaryDirectory
import shutil
from tensorflow.keras.losses import Huber

#path 설정
font_path = 'C:/Users/HOME/Desktop/Program/BMJUA_ttf.ttf'

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__))) if hasattr(sys, '_MEIPASS') else os.path.abspath(".")
    return os.path.join(base_path, relative_path)

def load_custom_font(font_path):
    """ Load a custom TTF font and return the font name """
    with TemporaryDirectory() as tempdir:
        temp_font_path = os.path.join(tempdir, os.path.basename(font_path))
        shutil.copy(font_path, temp_font_path)
        font = TTFont(temp_font_path)
        font_name = font['name'].getName(1, 3, 1, 1033).toUnicode()
        return font_name

# 모델과 스케일러 로드를 위한 함수
def load_models_and_scalers(frequencies):
    models = {}
    scalers = {}
    for freq in frequencies:
        model_path = os.path.join('C:/Users/HOME/Desktop/Program/DNN MODEL', f'TR_model_{freq}.h5')
        scaler_x_path = os.path.join('C:/Users/HOME/Desktop/Program/DNN MODEL', f'scaler_x_{freq}.pkl')
        scaler_y_path = os.path.join('C:/Users/HOME/Desktop/Program/DNN MODEL', f'scaler_y_{freq}.pkl')
        models[freq] = tf.keras.models.load_model(model_path, compile=False)
        scalers[freq] = {
            'x': joblib.load(scaler_x_path),
            'y': joblib.load(scaler_y_path)
        }
    return models, scalers

# 탐색 범위 제한
def clamp_parameters(N, L1, L2, W, S):
    N = max(0.25, min(N, 5))
    L1 = max(2, min(L1, 15))
    L2 = max(2, min(L2, 15))
    W = max(0.05, min(W, 1))
    S = max(0.05, min(S, 1))
    return N, L1, L2, W, S

# 입력 값에 대해 모델 예측을 수행하는 함수
def predict_model(model, input_data, scaler_x, scaler_y):
    input_data_scaled = scaler_x.transform(input_data)
    predictions = model.predict(input_data_scaled)
    predicted_values = scaler_y.inverse_transform(predictions)

    N = round(predicted_values[0, 0] * 4) / 4
    L1, L2, W, S = np.round(predicted_values[0, 1:], 2)

    # 최소 단위로 변환 및 범위 제한
    L1 = round(L1 * 10) / 10
    L2 = round(L2 * 10) / 10
    W = round(W * 100) / 100
    S = round(S * 100) / 100
    N, L1, L2, W, S = clamp_parameters(N, L1, L2, W, S)

    return N, L1, L2, W, S

# 예측 실행 함수
def run_predictions(models, scalers, L, Q, progress_var, progress_label):
    input_data = np.array([[L, Q]], dtype=np.float32)
    results = []
    total_freqs = len(models)
    huber_loss = Huber()

    for i, (freq, model) in enumerate(models.items()):
        scaler = scalers[freq]
        scaler_x = scaler['x']
        scaler_y = scaler['y']
        
        try:
            N, L1, L2, W, S = predict_model(model, input_data, scaler_x, scaler_y)
            # Loss 계산
            y_true = np.array([[L, Q, N, L1, L2, W, S]])
            y_pred = np.array([[L, Q, N, L1, L2, W, S]])
            loss = huber_loss(y_true, y_pred).numpy()
            results.append((freq, N, L1, L2, W, S, loss))

        except Exception as e:
            print(f"Error during prediction for {freq}: {e}")
            continue

        progress = (i + 1) / total_freqs * 100
        progress_var.set(progress)
        progress_label.config(text=f"{int(progress)}%")
        root.update_idletasks()
    
    progress_label.config(text="Complete")
    # 주파수 기준 오름차순 정렬
    results.sort(key=lambda x: float(x[0].replace('MHz', '')))
    return results

# 모델 로드 버튼 클릭 시 실행 함수
def load_models():
    selected_frequencies = [freq for freq, var in frequency_vars.items() if var.get()]
    if not selected_frequencies:
        status_label.config(text="No frequency band selected", fg="red")
        return
    
    try:
        global models, scalers
        models, scalers = load_models_and_scalers(selected_frequencies)
        status_label.config(text="Models and scalers loaded successfully", fg="green")
        load_button.config(state=tk.DISABLED)
        predict_button.config(state=tk.NORMAL)
        reset_button.config(state=tk.NORMAL)  # Reset 버튼 활성화
        for cb in checkbuttons:
            cb.config(state=tk.DISABLED)
    except Exception as e:
        status_label.config(text=f"Failed to load models: {e}", fg="red")

# 최적화 버튼 클릭 시 실행 함수
def optimize_button_clicked():
    try:
        L = float(L_entry.get().replace("[nH]", "").strip())
        Q = float(Q_entry.get())
    except ValueError:
        messagebox.showerror("Error", "Please enter valid numbers for L and Q.")
        return

    progress_var.set(0)
    progress_label.config(text="0%")
    root.update_idletasks()

    results = run_predictions(models, scalers, L, Q, progress_var, progress_label)

    for row in tree.get_children():
        tree.delete(row)
    
    for result in results:
        freq, N, L1, L2, W, S, loss = result
        tree.insert("", tk.END, values=(freq, N, f"{L1:.2f} [mm]", f"{L2:.2f} [mm]", f"{W:.2f} [mm]", f"{S:.2f} [mm]", f"{loss:.4f}"))

# 폰트 크기 조절 슬라이더 콜백 함수
def adjust_font_size(event):
    size = int(font_size_slider.get())
    custom_font.configure(size=size)
    title_font.configure(size=int(size * 1.5))

# 모두 선택/선택 해제 버튼 콜백 함수
def toggle_select_all():
    select_all = all(var.get() for var in frequency_vars.values())
    for var in frequency_vars.values():
        var.set(not select_all)

# 선택 초기화 버튼 클릭 시 실행 함수
def reset_selection():
    for var in frequency_vars.values():
        var.set(False)
    load_button.config(state=tk.NORMAL)
    predict_button.config(state=tk.DISABLED)
    status_label.config(text="Please Select Frequency", fg="black")
    for cb in checkbuttons:
        cb.config(state=tk.NORMAL)

# L 입력 박스에 포커스 얻기/잃기 이벤트 핸들러
def on_L_focus_in(event):
    if L_entry.get() == "(Inductance [nH])":
        L_entry.delete(0, tk.END)
        L_entry.configure(foreground='black')
    elif L_entry.get().endswith("[nH]"):
        L_entry.configure(foreground='black')

def on_L_focus_out(event=None):
    if not L_entry.get():
        L_entry.insert(0, "(Inductance [nH])")
        L_entry.configure(foreground='grey')
    elif not L_entry.get().endswith("[nH]"):
        current_text = L_entry.get()
        L_entry_var.set(f"{current_text} [nH]")

# Q 입력 박스에 포커스 얻기/잃기 이벤트 핸들러
def on_Q_focus_in(event):
    if Q_entry.get() == "(Quality factor)":
        Q_entry.delete(0, tk.END)
        Q_entry.configure(foreground='black')

def on_Q_focus_out(event=None):
    if not Q_entry.get():
        Q_entry.insert(0, "(Quality factor)")
        Q_entry.configure(foreground='grey')

# 헬프 버튼 클릭 시 실행 함수
def show_help():
    help_window = tk.Toplevel(root)
    help_window.title("Help")

    tab_control = ttk.Notebook(help_window)
    introduction_tab = ttk.Frame(tab_control)
    usage_tab = ttk.Frame(tab_control)
    parameter_tab = ttk.Frame(tab_control)
    creator_tab = ttk.Frame(tab_control)
    
    tab_control.add(introduction_tab, text='Introduction')
    tab_control.add(usage_tab, text='User Manual')
    tab_control.add(parameter_tab, text='Parameters')
    tab_control.add(creator_tab, text='Credit')
    
    tab_control.pack(expand=1, fill='both')
    
    # 서문
    introduction_text = tk.Text(introduction_tab, wrap='word', height=15, width=50)
    introduction_text.insert(tk.END, "Spiral Inductor Design Optimization Project using deep neural network model\n\n")
    introduction_text.insert(tk.END, "This program is designed to predict the minimum size of an inductor with desired specifications.\n")
    introduction_text.insert(tk.END, "When the user inputs the required specifications, the physical values of the rectangular planar spiral inductor are output.\n")
    introduction_text.insert(tk.END, "The parameters output at this time are the parameters of the inductor with the minimum L1*L2 value.\n")
    introduction_text.insert(tk.END, "This program was created using a deep neural network model with Python.\n")
    introduction_text.config(state=tk.DISABLED)
    introduction_text.pack(padx=10, pady=10)

    # 사용법
    usage_text = tk.Text(usage_tab, wrap='word', height=15, width=50)
    usage_text.insert(tk.END, "Here are the instructions on how to use the application...\n\n")
    usage_text.insert(tk.END, "1. Select Frequency: Choose the frequencies you want to use.\n")
    usage_text.insert(tk.END, "2. Done: Load the selected models.\n")
    usage_text.insert(tk.END, "3. Reset Selection: Reset the frequency selections.\n")
    usage_text.insert(tk.END, "4. Input Parameter: Enter the values for L and Q.\n")
    usage_text.insert(tk.END, "5. Start Predict: Run the prediction.\n")
    usage_text.config(state=tk.DISABLED)
    usage_text.pack(padx=10, pady=10)
    
    # 파라미터 설명
    parameter_text = tk.Text(parameter_tab, wrap='word', height=15, width=50)
    parameter_text.insert(tk.END, "Explanation of parameters...\n\n")
    parameter_text.insert(tk.END, "L (Inductance): The inductance value in nanohenries (nH).\n")
    parameter_text.insert(tk.END, "Q (Quality Factor): The quality factor of the inductor, which indicates the efficiency of the inductor.\n")
    parameter_text.insert(tk.END, "N (Number of Turns): The number of complete loops that the inductor coil makes.\n")
    parameter_text.insert(tk.END, "L1 and L2 (Length 1 and Length 2): The lengths of the sides of the rectangular inductor.\n")
    parameter_text.insert(tk.END, "W (Width): The width of the traces of the inductor.\n")
    parameter_text.insert(tk.END, "S (Spacing): The spacing between the traces of the inductor.\n")
    parameter_text.config(state=tk.DISABLED)
    parameter_text.pack(padx=10, pady=10)
    
    # 제작자
    creator_text = tk.Text(creator_tab, wrap='word', height=15, width=50)
    creator_text.insert(tk.END, "Information about the creator...\n\n")
    creator_text.insert(tk.END, "Department of Electronic Convergence Engineering, Kwangwoon University\n")
    creator_text.config(state=tk.DISABLED)
    creator_text.pack(padx=10, pady=10)

# GUI 생성
root = tk.Tk()
root.title("Planar Inductor Designer")

# 폰트 로드 및 초기 설정
custom_font_name = load_custom_font(font_path)
custom_font = font.Font(family=custom_font_name, size=14)
title_font = font.Font(family=custom_font_name, size=int(14 * 1.5))
root.option_add('*Font', custom_font)

# 창 크기 설정
root.geometry("600x1000")

# 스크롤 캔버스 설정
canvas = tk.Canvas(root)
scroll_y = tk.Scrollbar(root, orient="vertical", command=canvas.yview)

scroll_frame = tk.Frame(canvas)
scroll_frame.bind(
    "<Configure>",
    lambda e: canvas.configure(
        scrollregion=canvas.bbox("all")
    )
)

canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
canvas.configure(yscrollcommand=scroll_y.set)

canvas.pack(side="left", fill="both", expand=True)
scroll_y.pack(side="right", fill="y")

# 전체 가운데 정렬을 위한 프레임
center_frame = tk.Frame(scroll_frame)
center_frame.pack(expand=True)

# 주파수 선택 레이블
frequency_label = tk.Label(center_frame, text="Select Frequency", font=title_font)
frequency_label.grid(row=0, column=0, pady=10)

# 주파수 대역 선택 체크박스
frequency_options = [f'{i}.0MHz' for i in range(100, 1000, 100)]
frequency_vars = {freq: tk.BooleanVar() for freq in frequency_options}

checkbuttons_frame = tk.Frame(center_frame)
checkbuttons_frame.grid(row=1, column=0, pady=10)

checkbuttons = []
for i, (freq, var) in enumerate(frequency_vars.items()):
    cb = tk.Checkbutton(checkbuttons_frame, text=freq, variable=var)
    cb.grid(row=i // 3, column=i % 3, padx=10, pady=5)
    checkbuttons.append(cb)

# 모두 선택/선택 해제 버튼
select_all_button = tk.Button(center_frame, text="Select All/Unselect All", command=toggle_select_all)
select_all_button.grid(row=2, column=0, pady=10)

# 두 버튼 중앙 배치
button_frame = tk.Frame(center_frame)
button_frame.grid(row=3, column=0, pady=10)
# 모델 로드 버튼
load_button = tk.Button(button_frame, text="Done", command=load_models)
load_button.grid(row=0, column=0, padx=(10, 5))
# 선택 초기화 버튼
reset_button = tk.Button(button_frame, text="Reset Selection", command=reset_selection, state=tk.DISABLED)
reset_button.grid(row=0, column=1, padx=(5, 10))

# 상태 레이블
status_label = tk.Label(center_frame, text="Please Select Frequency", font=custom_font)
status_label.grid(row=4, column=0, pady=10)

# 입력 파라미터 레이블
input_param_label = tk.Label(center_frame, text="Input Parameter", font=title_font)
input_param_label.grid(row=5, column=0, pady=10)

# L과 Q 입력
input_frame = tk.Frame(center_frame)
input_frame.grid(row=6, column=0, pady=5)

L_entry_var = tk.StringVar()
L_entry_var.set("(Inductance [nH])")
L_entry = tk.Entry(input_frame, width=20, textvariable=L_entry_var, foreground='grey', justify='center')  # 입력 박스 가운데 정렬
L_entry.bind("<FocusIn>", on_L_focus_in)
L_entry.bind("<FocusOut>", on_L_focus_out)
L_entry.grid(row=0, column=0, padx=5)

Q_entry_var = tk.StringVar()
Q_entry_var.set("(Quality factor)")
Q_entry = tk.Entry(input_frame, width=20, textvariable=Q_entry_var, foreground='grey', justify='center')  # 입력 박스 가운데 정렬
Q_entry.bind("<FocusIn>", on_Q_focus_in)
Q_entry.bind("<FocusOut>", on_Q_focus_out)
Q_entry.grid(row=0, column=1, padx=5)

# 두 버튼 중앙 배치
opt_frame = tk.Frame(center_frame)
opt_frame.grid(row=7, column=0, pady=10)
# 예측 버튼
predict_button = tk.Button(opt_frame, text="Start Predict", command=optimize_button_clicked, state=tk.DISABLED)
predict_button.grid(row=0, column=0, padx=(5, 10))

# 진행 상황 게이지
progress_frame = tk.Frame(center_frame)
progress_frame.grid(row=8, column=0, pady=10, padx=10)

progress_var = tk.DoubleVar()
progress_bar = ttk.Progressbar(progress_frame, variable=progress_var, maximum=100)
progress_bar.pack(fill=tk.X, expand=1)

progress_label = tk.Label(progress_frame, text="0%")
progress_label.pack()

# 결과 타이틀 레이블
result_label = tk.Label(center_frame, text="Predict Result", font=title_font)
result_label.grid(row=9, column=0, pady=10)

# 결과 표
columns = ("Freq", "N", "L1", "L2", "W", "S")
tree = ttk.Treeview(center_frame, columns=columns, show="headings", height=10)

# 각 열의 제목과 정렬 설정
for col in columns:
    tree.heading(col, text=col, anchor=tk.CENTER)
    tree.column(col, anchor=tk.CENTER, width=100)

tree.grid(row=10, column=0, pady=10, sticky="nsew")
center_frame.grid_rowconfigure(10, weight=1)
center_frame.grid_columnconfigure(0, weight=1)

# 폰트 크기 조절 슬라이더
font_size_frame = tk.Frame(center_frame)
font_size_frame.grid(row=11, column=0, sticky="w", padx=10, pady=10)

font_size_slider = ttk.Scale(font_size_frame, from_=8, to_=48, orient='horizontal', command=adjust_font_size)
font_size_slider.set(14)
font_size_slider.pack(side=tk.LEFT)

# Help 버튼
help_button = tk.Button(center_frame, text="Help", command=show_help)
help_button.grid(row=12, column=0, sticky="e", padx=10)

# 스크롤 이벤트 핸들러
def _on_mousewheel(event):
    canvas.yview_scroll(int(-1*(event.delta/120)), "units")

canvas.bind_all("<MouseWheel>", _on_mousewheel)

# GUI 실행
root.mainloop()
