import PySimpleGUI as sg
import configparser
import time

from aitextgen import aitextgen
from transformers import GPT2Tokenizer
import torch
import cpuinfo
from psutil import virtual_memory

sg.theme('Dark Blue 3')

ai = None
tokenizer = None
use_fp16 = False
gpubool = None
showgpustuff = None

model_info = {'GPT-Neo 125M': 2, 'GPT-Neo 1.3B': 8, 'GPT-Neo 2.7B': 12, 'GPT-2 124M': 1, 'GPT-2 355M': 4, 'GPT-2 774M': 6, 'GPT-2 1558M': 10, 'model_type': {'GPT-Neo 125M': 'non_gpt2', 'GPT-Neo 1.3B': 'non_gpt2', 'GPT-Neo 2.7B': 'non_gpt2', 'GPT-2 124M': 'tf_gpt2', 'GPT-2 355M': 'tf_gpt2', 'GPT-2 774M': 'tf_gpt2', 'GPT-2 1558M': 'tf_gpt2', 'nongpt2': {'GPT-Neo 125M': 'EleutherAI/gpt-neo-125M', 'GPT-Neo 1.3B': 'EleutherAI/gpt-neo-1.3B', 'GPT-Neo 2.7B': 'EleutherAI/gpt-neo-2.7B'}, 'tfgpt2': {'GPT-2 124M': '124M', 'GPT-2 355M': '355M', 'GPT-2 774M': '774M', 'GPT-2 1558M': '1558M'}}}

menu_def = [['&File', ['COULD DO MORE STUFF HERE', '&Options', 'E&xit']],
            ['&Help', ['&About']]]

data = [["INPUT", "OUTPUT", "TOKENCOUNT"], ["INPUT2", "OUTPUT2", "TOKENCOUNT2"], ["INPUT3", "OUTPUT3", "TOKENCOUNT3"], ["INPUT4", "OUTPUT4", "TOKENCOUNT4"], ["INPUT5", "OUTPUT5", "TOKENCOUNT5"], ["INPUT6", "OUTPUT6", "TOKENCOUNT6"], ["INPUT7", "OUTPUT7", "TOKENCOUNT7"], ["INPUT8", "OUTPUT8", "TOKENCOUNT8"], ["Line1\nLine2\nLine3", "Line1\nLine2\nLine3", "Line1\nLine2\nLine3"]]
headings = ["Input", "Output", "Token Count"]

side_buttons_table = [[sg.Text('Fewshot List Options')],
                      [sg.Button('Display selected pair')], # This should generate a popup asking if they're sure - it'll wipe whatever they've already got in there. We don't need to show the popup if they don't have anything in the boxes, though.
                      [sg.Button('Remove selected pair(s)')],
                      [sg.Button('Save fewshots to file')],
                      [sg.Button('Load fewshots from file')],
                      [sg.Button('Clear fewshot table')]]

side_buttons_input = [[sg.Text('Current Pair Options')],
                      [sg.Button('Save pair to table')],
                      [sg.Button('Regen output')],
                      [sg.Button('Preview trimmed pairs')],
                      [sg.Button('Clear input and output')]]

main_layout = [[sg.Menu(menu_def)],
               [sg.Button('Change input prefix'), sg.Button('Change output prefix'), sg.Button('Export')],
               [sg.Table(values=data, headings=headings, max_col_width=100,
                                background_color='darkblue',
                                auto_size_columns=True,
                                justification='center',
                                num_rows=5,
                                alternating_row_color='darkblue',
                                key='-TABLE-',
                                expand_x=True,
                                row_height=100), sg.Col(side_buttons_table, justification='right', vertical_alignment='top')],
               [sg.Text('Input')],
               [sg.Multiline('', size=(100,10)), sg.Col(side_buttons_input, justification='right', vertical_alignment='top')],
               [sg.Text('Output')],
               [sg.Multiline('', size=(100,10))]]

model_window_layout = [[sg.Text('Test Window')],
                       [sg.Button('OK')]]

# window = sg.Window('GPT Fewshot Batcher', main_layout, location=(0,0))

""" event, values = window.read()
event, values = model_window.read() """
# window.close()

def first_boot():
    global ai, tokenizer, use_fp16, gpubool, showgpustuff
    if torch.cuda.is_available():
        gpusupport = 'YES'
        showgpustuff = True
        devicename = torch.cuda.get_device_name()
        deviceramtext = 'GPU VRAM: '
        deviceram = str(round(torch.cuda.get_device_properties(0).total_memory / (1024.0 **3)))
        devicecolor = 'lightgreen'
        gpubool = True
    else:
        gpusupport = 'NO'
        showgpustuff = False
        devicename = f"Will run on {str(cpuinfo.get_cpu_info()['brand_raw'])} instead"
        deviceramtext = 'System RAM: '
        deviceram = str(round(virtual_memory().total / (1024.0 **3)))
        devicecolor = 'orange'
        gpubool = False

    layout = [[sg.Text("First boot detected! It is recommended that you select and download an AI model before continuing.")],
              [sg.Text("GPU detected:"), sg.Text(f"{gpusupport} - {devicename}", text_color = devicecolor, key='-GPUSUPPORTTEXT-')],
              [sg.Text(deviceramtext, key='-DEVICERAMPREFIX-'), sg.Text(f"{deviceram} GB", text_color = devicecolor, key='-DEVICERAMTEXT-')],
              [sg.Checkbox('GPU Enabled', default=showgpustuff, visible=showgpustuff, key='-GPUCHECKBOX-', enable_events=True)],
              [sg.Text("Available models:")],
              [sg.Combo(['No model', 'GPT-Neo 125M', 'GPT-Neo 1.3B', 'GPT-Neo 2.7B', 'GPT-2 124M', 'GPT-2 355M', 'GPT-2 774M', 'GPT-2 1558M'], key='-MODEL-', enable_events=True), sg.Checkbox('FP16', default=use_fp16, visible=False, key='-FP16CHECKBOX-', enable_events=True)],
              [sg.Text("Great! You should be able to run this model!", text_color = 'lightgreen', visible=False, key='-CANRUNMODEL-'), sg.Text("Uh oh... it looks like you don't meet the minimum memory requirements for this model. You can still try to run it, but it may not work.", text_color = 'orange', visible=False, key='-CANTRUNMODEL-')],
              [sg.Column([[sg.Button("Select & Download", key='-SELECT-', visible=False), sg.Button("Exit", key='-EXIT-')]], justification='center', vertical_alignment='top')]]
    window = sg.Window("Model Selection", layout, modal=True)
    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED:
            break
        if values['-MODEL-'] and not values['-MODEL-'] == 'No model':
            window['-SELECT-'].update(visible=True)
            if gpubool == True:
                window['-FP16CHECKBOX-'].update(visible=True)
            if int(deviceram) >= model_info[values['-MODEL-']] or (values['-FP16CHECKBOX-'] == True and int(deviceram) >= model_info[values['-MODEL-']] // 2):
                window['-CANTRUNMODEL-'].update(visible=False)
                window['-CANRUNMODEL-'].update(visible=True)
            else:
                window['-CANTRUNMODEL-'].update(visible=True)
                window['-CANRUNMODEL-'].update(visible=False)
        else:
            window['-CANTRUNMODEL-'].update(visible=False)
            window['-CANRUNMODEL-'].update(visible=False)
            window['-SELECT-'].update(visible=False)
            window['-FP16CHECKBOX-'].update(visible=False)
        if event == '-GPUCHECKBOX-' and showgpustuff == True:
            if values['-GPUCHECKBOX-'] == False:
                gpusupport = 'NO'
                devicename = f"Will run on {str(cpuinfo.get_cpu_info()['brand_raw'])} instead"
                deviceramtext = 'System RAM: '
                deviceram = str(round(virtual_memory().total / (1024.0 **3)))
                devicecolor = 'red'
                gpubool = False
                use_fp16 = False
                window['-FP16CHECKBOX-'].update(visible=False)
            else:
                gpusupport = 'YES'
                devicename = torch.cuda.get_device_name()
                deviceramtext = 'GPU VRAM: '
                deviceram = str(round(torch.cuda.get_device_properties(0).total_memory / (1024.0 **3)))
                devicecolor = 'lightgreen'
                gpubool = True
                if not values['-MODEL-'] == '' and not values['-MODEL-'] == 'No model':
                    window['-FP16CHECKBOX-'].update(visible=True)
            if values['-MODEL-'] and not values['-MODEL-'] == 'No model':
                if int(deviceram) >= model_info[values['-MODEL-']] or (values['-FP16CHECKBOX-'] == True and int(deviceram) >= model_info[values['-MODEL-']] // 2):
                    window['-CANTRUNMODEL-'].update(visible=False)
                    window['-CANRUNMODEL-'].update(visible=True)
                else:
                    window['-CANTRUNMODEL-'].update(visible=True)
                    window['-CANRUNMODEL-'].update(visible=False)
            else:
                window['-CANTRUNMODEL-'].update(visible=False)
                window['-CANRUNMODEL-'].update(visible=False)
            window['-DEVICERAMPREFIX-'].update(deviceramtext)
            window['-GPUSUPPORTTEXT-'].update(f"{gpusupport} - {devicename}", text_color = devicecolor)
            window['-DEVICERAMTEXT-'].update(f"{deviceram} GB", text_color = devicecolor)
        if event == '-FP16CHECKBOX-':
            use_fp16 = values['-FP16CHECKBOX-']
        if event == '-SELECT-' and not values['-MODEL-'] == 'No model':
            if sg.popup_yes_no(f"Are you sure you want to download the {values['-MODEL-']} model?", title="Confirm Model Selection", keep_on_top=True, modal=True) == 'Yes':
                if model_info['model_type'][values['-MODEL-']] == 'tf_gpt2':
                    ai = aitextgen(tf_gpt2=model_info['model_type']['tfgpt2'][values['-MODEL-']], to_gpu=gpubool, to_fp16=use_fp16, cache_dir=f"./models/gpt2-{model_info['model_type']['tfgpt2'][values['-MODEL-']]}")
                    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
                else:
                    ai = aitextgen(model=model_info['model_type']['nongpt2'][values['-MODEL-']], to_gpu=gpubool, to_fp16=use_fp16, cache_dir=f"./models/{model_info['model_type']['nongpt2'][values['-MODEL-']]}")
                    tokenizer = GPT2Tokenizer.from_pretrained(model_info['model_type']['nongpt2'][values['-MODEL-']])
                break
        if event == '-EXIT-' and (values['-MODEL-'] == 'No model' or not values['-MODEL-']):
            if sg.popup_yes_no('Are you sure you want to use GPT Fewshot Batcher with no model selected and downloaded? You won\'t be able to generate or tokenize anything!', title="Confirm No Model", keep_on_top = True) == 'Yes':
                break
    window.close()

def main():
    try:
        configfile = open("config.ini", 'r+')
        # window = sg.Window('GPT Fewshot Batcher', main_layout, location=(0,0))
    except FileNotFoundError:
        # configfile = open("config.ini", 'w+')
        first_boot()
        print('WE DID IT')
        window = sg.Window('GPT Fewshot Batcher', main_layout, location=(0,0))
        event, values = window.read()
        window.close()

if __name__ == "__main__":
    main()