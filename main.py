import PySimpleGUI as sg
import configparser

from aitextgen import aitextgen
import torch
import cpuinfo
from psutil import virtual_memory

sg.theme('Dark Blue 3')

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

window = sg.Window('GPT Fewshot Batcher', main_layout, location=(0,0))

""" event, values = window.read()
event, values = model_window.read() """
window.close()

def first_boot():
    if torch.cuda.is_available():
        gpusupport = 'YES'
        showgpustuff = True
        devicename = torch.cuda.get_device_name()
        deviceramtext = 'GPU VRAM: '
        deviceram = str(round(torch.cuda.get_device_properties(0).total_memory / (1024.0 **3)))
        devicecolor = 'lightgreen'
    else:
        gpusupport = 'NO'
        showgpustuff = False
        devicename = 'Will run on '+str(cpuinfo.get_cpu_info()['brand_raw'])+' instead'
        deviceramtext = 'System RAM: '
        deviceram = str(round(virtual_memory().total / (1024.0 **3)))
        devicecolor = 'red'

    layout = [[sg.Text("First boot detected! It is recommended that you select and download an AI model before continuing.")],
              [sg.Text("GPU detected:"), sg.Text(gpusupport+" - "+devicename, text_color = devicecolor, key='-GPUSUPPORTTEXT-')],
              [sg.Text(deviceramtext, key='-DEVICERAMPREFIX-'), sg.Text(deviceram+" GB", text_color = devicecolor, key='-DEVICERAMTEXT-')],
              [sg.Checkbox('GPU Enabled', default=showgpustuff, visible=showgpustuff, key='-GPUCHECKBOX-', enable_events=True)],
              [sg.Text("Available models:")],
              [sg.Combo(['No model', 'GPT-Neo 125M', 'GPT-Neo 1.3B', 'GPT-Neo 2.7B', 'GPT-2 124M', 'GPT-2 355M', 'GPT-2 774M', 'GPT-2 1558M'], key='-MODEL-', enable_events=True)],
              [sg.Column([[sg.Button("Select & Download", key='-SELECT-', visible=False), sg.Button("Exit", key='-EXIT-')]], justification='center', vertical_alignment='top')]]
    window = sg.Window("Model Selection", layout, modal=True)
    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED:
            break
        if values['-MODEL-'] and not values['-MODEL-'] == 'No model':
            window['-SELECT-'].update(visible=True)
        else:
            window['-SELECT-'].update(visible=False)
        if event == '-GPUCHECKBOX-' and showgpustuff == True:
            if values['-GPUCHECKBOX-'] == False:
                gpusupport = 'NO'
                devicename = 'Will run on '+str(cpuinfo.get_cpu_info()['brand_raw'])+' instead'
                deviceramtext = 'System RAM: '
                deviceram = str(round(virtual_memory().total / (1024.0 **3)))
                devicecolor = 'red'
            else:
                gpusupport = 'YES'
                devicename = torch.cuda.get_device_name()
                deviceramtext = 'GPU VRAM: '
                deviceram = str(round(torch.cuda.get_device_properties(0).total_memory / (1024.0 **3)))
                devicecolor = 'lightgreen'
            window['-DEVICERAMPREFIX-'].update(deviceramtext)
            window['-GPUSUPPORTTEXT-'].update(gpusupport+" - "+devicename, text_color = devicecolor)
            window['-DEVICERAMTEXT-'].update(deviceram+" GB", text_color = devicecolor)
        if event == '-EXIT-' and (values['-MODEL-'] == 'No model' or not values['-MODEL-']):
            if sg.popup_yes_no('Are you sure you want to use GPT Fewshot Batcher with no model selected and downloaded? You won\'t be able to generate or tokenize anything!', title="Confirm No Model", keep_on_top = True) == 'Yes':
                print('okay then')
    window.close()

def main():
    try:
        configfile = open("config.ini", 'r+')
    except FileNotFoundError:
        # configfile = open("config.ini", 'w+')
        first_boot()

if __name__ == "__main__":
    main()