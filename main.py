import PySimpleGUI as sg
import platform

from aitextgen import aitextgen
from transformers import GPT2Tokenizer
import torch
import cpuinfo
from psutil import virtual_memory

sg.theme('Dark Blue 3')

def initialize_config():
    config = sg.UserSettings(filename='config.ini', path='.')
    config['use_fp16'] = False
    config['gpubool'] = None
    config['nomodel'] = None
    config['defaultmodel'] = None
    config['model_type'] = None
    config['model_length'] = 128
    config['model_temp'] = 0.9
    config['model_rep_pen'] = 1.0
    config['model_length_pen'] = 1.0
    config['model_top_k'] = 50
    config['model_top_p'] = 1.0
    config['model_inputprefix'] = 'Input:'
    config['model_outputprefix'] = 'Output:'
    return config

model_info = {'GPT-Neo 125M': 2, 'GPT-Neo 1.3B': 8, 'GPT-Neo 2.7B': 12, 'GPT-2 124M': 1, 'GPT-2 355M': 4, 'GPT-2 774M': 6, 'GPT-2 1558M': 10, 'model_type': {'GPT-Neo 125M': 'non_gpt2', 'GPT-Neo 1.3B': 'non_gpt2', 'GPT-Neo 2.7B': 'non_gpt2', 'GPT-2 124M': 'tf_gpt2', 'GPT-2 355M': 'tf_gpt2', 'GPT-2 774M': 'tf_gpt2', 'GPT-2 1558M': 'tf_gpt2', 'nongpt2': {'GPT-Neo 125M': 'EleutherAI/gpt-neo-125M', 'GPT-Neo 1.3B': 'EleutherAI/gpt-neo-1.3B', 'GPT-Neo 2.7B': 'EleutherAI/gpt-neo-2.7B'}, 'tfgpt2': {'GPT-2 124M': '124M', 'GPT-2 355M': '355M', 'GPT-2 774M': '774M', 'GPT-2 1558M': '1558M'}}}

def main_window(config, ai, tokenizer):
    menu_def = [['&File', ['COULD DO MORE STUFF HERE', '&Settings', 'E&xit']],
            ['&Help', ['&About']]]

    tabledisplay = [['', '', ''], ['', '', ''], ['', '', ''], ['', '', ''], ['', '', ''], ['', '', '']]
    tabledata = []
    assembled_context = ''
    headings = ["Input", "Output", "Token Count"]

    side_buttons_table = [[sg.Text('Fewshot List Options')],
                          [sg.Button('Display selected pair')], # This should generate a popup asking if they're sure - it'll wipe whatever they've already got in there. We don't need to show the popup if they don't have anything in the boxes, though.
                          [sg.Button('Preview trimmed pair(s)')],
                          [sg.Button('Remove selected pair(s)')],
                          [sg.Button('Save fewshots to file')],
                          [sg.Button('Load fewshots from file')],
                          [sg.Button('Clear fewshot table')]]

    side_buttons_input = [[sg.Text('Current Pair Options')],
                          [sg.Button('Save pair to table', key='-SAVEPAIR-')],
                          [sg.Button('(Re)generate output', key='-GENERATE-')],
                          [sg.Button('Clear input and output', key='-CLEAR-')],
                          [sg.Button('Testing button!', key='-TESTING-')]]

    main_layout = [[sg.Menu(menu_def)],
                   [sg.Button('Change input prefix', key='-INPUTPREFIX-'), sg.Button('Change output prefix', key='-OUTPUTPREFIX-'), sg.Button('Export'), sg.Button('Settings'), sg.Button('Maybe have a text box where all the formatted text goes?')],
                   [sg.Table(values=tabledisplay, headings=headings, max_col_width=100,
                                    background_color='darkblue',
                                    auto_size_columns=True,
                                    justification='center',
                                    num_rows=min(len(tabledisplay), 1000),
                                    alternating_row_color='darkblue',
                                    key='-TABLE-',
                                    expand_x=True,
                                    row_height=100), sg.Col(side_buttons_table, justification='right', vertical_alignment='top')],
                   [sg.Text('Input')],
                   [sg.Multiline('', size=(100,10), key='-INPUTBOX-'), sg.Col(side_buttons_input, justification='right', vertical_alignment='top')],
                   [sg.Text('Output')],
                   [sg.Multiline('', size=(100,10), key='-OUTPUTBOX-')]]

    window = sg.Window('GPT Fewshot Batcher', main_layout, location=(0,0))
    
    while True:
        event, values = window.read()
        # Test Linux!!!!!
        def newline_fix(try_strip):
            if platform.system() == 'Darwin':
                stripped_fix = try_strip.rstrip(try_strip[-1])
                return stripped_fix
            else:
                return try_strip
        def assemble_context(assembled_context):
            for index, value in enumerate(tabledata):
                if index == 0:
                    assembled = f"{config['model_inputprefix']}\n\n{value['input']}\n\n{config['model_outputprefix']}\n\n{value['output']}"
                else:
                    assembled = f"\n\n{config['model_inputprefix']}\n\n{value['input']}\n\n{config['model_outputprefix']}\n\n{value['output']}"
                assembled_context = f"{assembled_context}{assembled}"
            return assembled_context
        def generate_text(assemble, assembled_context):
            if assemble:
                assembled_context = assemble_context(assembled_context)
                prompt_temp = f"{assembled_context}\n\n{config['model_inputprefix']}\n\n{newline_fix(values['-INPUTBOX-'])}\n\n{config['model_outputprefix']}"
            else:
                prompt_temp = f"{config['model_inputprefix']}\n\n{newline_fix(values['-INPUTBOX-'])}\n\n{config['model_outputprefix']}"
            prompt_tokens = tokenizer.encode(prompt_temp)
            maxlen = config['model_length'] + len(prompt_tokens)
            gen_text = ai.generate_one(prompt = prompt_temp,
                                       min_length = len(prompt_tokens)+1,
                                       max_length = maxlen,
                                       temperature = config['model_temp'],
                                       repetition_penalty = config['model_rep_pen'],
                                       length_penalty = config['model_length_pen'],
                                       top_k = config['model_top_k'],
                                       top_p = config['model_top_p']
                                       )
            try:
                gen_stripped_text = gen_text[len(prompt_temp)+2:].split(f"\n\n{config['model_inputprefix']}", 1)[0]
            except:
                gen_stripped_text = gen_text[len(prompt_temp)+2:]
            window['-OUTPUTBOX-'].update(gen_stripped_text)
        def tokenize_single_fewshot():
            if tabledisplay[0] == ['', '', ''] or len(tabledisplay) == 0:
                assembled = f"{config['model_inputprefix']}\n\n{newline_fix(values['-INPUTBOX-'])}\n\n{config['model_outputprefix']}\n\n{newline_fix(values['-OUTPUTBOX-'])}"
            else:
                assembled = f"\n\n{config['model_inputprefix']}\n\n{newline_fix(values['-INPUTBOX-'])}\n\n{config['model_outputprefix']}\n\n{newline_fix(values['-OUTPUTBOX-'])}"
            assembled_tokens = tokenizer.encode(assembled)
            return assembled_tokens
        def tokenize_all_fewshots():
            for index, value in enumerate(tabledata):
                if index == 0:
                    assembled = f"{config['model_inputprefix']}\n\n{value['input']}\n\n{config['model_outputprefix']}\n\n{value['output']}"
                else:
                    assembled = f"\n\n{config['model_inputprefix']}\n\n{value['input']}\n\n{config['model_outputprefix']}\n\n{value['output']}"
                assembled_tokens = tokenizer.encode(assembled)
                value['tokens'] = len(assembled_tokens)
        # must do tabledisplay = update_table() in all uses
        def update_table():
            tabledisplay = [[x['input'], x['output'], x['tokens']] for x in tabledata]
            window['-TABLE-'].update(values=tabledisplay)
            return tabledisplay

        if event == sg.WIN_CLOSED:
            break
        if event == '-GENERATE-':
            if newline_fix(values['-INPUTBOX-']) == '':
                sg.popup_ok('Input box must have text!', title='Error')
            else:
                if tabledisplay[0] == ['', '', ''] or len(tabledisplay) == 0:
                    if sg.popup_yes_no('Are you sure you\'d like to generate text with an empty context?', title='Confirm generation') == 'Yes':
                        assemble = False
                        generate_text(assemble, assembled_context)
                else:
                    assemble = True
                    generate_text(assemble, assembled_context)
        if event == '-SAVEPAIR-':
            if newline_fix(values['-INPUTBOX-']) == '' or newline_fix(values['-OUTPUTBOX-']) == '':
                sg.popup_ok('Both text boxes must have text!', title='Error')
            else:
                # activated should be moved to config to support modes
                assembled_tokens = tokenize_single_fewshot()
                tempdict = {'input': newline_fix(values['-INPUTBOX-']), 'output': newline_fix(values['-OUTPUTBOX-']), 'tokens': len(assembled_tokens), 'activated': True, 'editing': False}
                tabledata.append(tempdict)
                tabledisplay = update_table()
                window['-INPUTBOX-'].update('')
                window['-OUTPUTBOX-'].update('')
        if event == '-INPUTPREFIX-':
            temp_inputprefix = sg.popup_get_text('Change the input prefix:',
                                                  title='Change input prefix',
                                                  default_text=config['model_inputprefix'])
            if temp_inputprefix is not None:
                config['model_inputprefix'] = temp_inputprefix
                if not tabledisplay[0] == ['', '', ''] and not len(tabledisplay) == 0:
                    tokenize_all_fewshots()
                    tabledisplay = update_table()
        if event == '-OUTPUTPREFIX-':
            temp_outputprefix = sg.popup_get_text('Change the output prefix:',
                                                  title='Change output prefix',
                                                  default_text=config['model_outputprefix'])
            if temp_outputprefix is not None:
                config['model_outputprefix'] = temp_outputprefix
                if not tabledisplay[0] == ['', '', ''] and not len(tabledisplay) == 0:
                    tokenize_all_fewshots()
                    tabledisplay = update_table()
        if event == '-CLEAR-':
            if sg.popup_yes_no('Are you sure?', title='Confirm clear') == 'Yes':
                window['-INPUTBOX-'].update('')
                window['-OUTPUTBOX-'].update('')
        if event == '-TESTING-':
            print('I don\'t do anything right now!')
    window.close()

def first_boot(config):
    if torch.cuda.is_available():
        gpusupport = 'YES'
        showgpustuff = True
        devicename = torch.cuda.get_device_name()
        deviceramtext = 'GPU VRAM: '
        deviceram = str(round(torch.cuda.get_device_properties(0).total_memory / (1024.0 **3)))
        devicecolor = 'lightgreen'
        config['gpubool'] = True
    else:
        gpusupport = 'NO'
        showgpustuff = False
        devicename = f"Will run on {str(cpuinfo.get_cpu_info()['brand_raw'])} instead"
        deviceramtext = 'System RAM: '
        deviceram = str(round(virtual_memory().total / (1024.0 **3)))
        devicecolor = 'orange'
        config['gpubool'] = False

    layout = [[sg.Text("First boot detected! It is recommended that you select and download an AI model before continuing.")],
              [sg.Text("GPU detected:"), sg.Text(f"{gpusupport} - {devicename}", text_color = devicecolor, key='-GPUSUPPORTTEXT-')],
              [sg.Text(deviceramtext, key='-DEVICERAMPREFIX-'), sg.Text(f"{deviceram} GB", text_color = devicecolor, key='-DEVICERAMTEXT-')],
              [sg.Checkbox('GPU Enabled', default=showgpustuff, visible=showgpustuff, key='-GPUCHECKBOX-', enable_events=True)],
              [sg.Text("Available models:")],
              [sg.Combo(['No model', 'GPT-Neo 125M', 'GPT-Neo 1.3B', 'GPT-Neo 2.7B', 'GPT-2 124M', 'GPT-2 355M', 'GPT-2 774M', 'GPT-2 1558M'], key='-MODEL-', enable_events=True), sg.Checkbox('FP16', default=config['use_fp16'], visible=False, key='-FP16CHECKBOX-', enable_events=True)],
              [sg.Text("Great! You should be able to run this model!", text_color = 'lightgreen', visible=False, key='-CANRUNMODEL-'), sg.Text("Uh oh... it looks like you don't meet the minimum memory requirements for this model. You can still try to run it, but it may not work.", text_color = 'orange', visible=False, key='-CANTRUNMODEL-')],
              [sg.Column([[sg.Button("Select & Download", key='-SELECT-', visible=False), sg.Button("Exit", key='-EXIT-')]], justification='center', vertical_alignment='top')]]
    window = sg.Window("Model Selection", layout, modal=True)
    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED:
            config['nomodel'] = True
            break
        if values['-MODEL-'] and not values['-MODEL-'] == 'No model':
            window['-SELECT-'].update(visible=True)
            if config['gpubool'] == True:
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
                config['gpubool'] = False
                config['use_fp16'] = False
                window['-FP16CHECKBOX-'].update(visible=False)
            else:
                gpusupport = 'YES'
                devicename = torch.cuda.get_device_name()
                deviceramtext = 'GPU VRAM: '
                deviceram = str(round(torch.cuda.get_device_properties(0).total_memory / (1024.0 **3)))
                devicecolor = 'lightgreen'
                config['gpubool'] = True
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
            config['use_fp16'] = values['-FP16CHECKBOX-']
        if event == '-SELECT-' and not values['-MODEL-'] == 'No model':
            if sg.popup_yes_no(f"Are you sure you want to download the {values['-MODEL-']} model?", title="Confirm Model Selection", keep_on_top=True, modal=True) == 'Yes':
                if model_info['model_type'][values['-MODEL-']] == 'tf_gpt2':
                    ai = aitextgen(tf_gpt2=model_info['model_type']['tfgpt2'][values['-MODEL-']], to_gpu=config['gpubool'], to_fp16=config['use_fp16'], cache_dir=f"./models/gpt2-{model_info['model_type']['tfgpt2'][values['-MODEL-']]}")
                    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
                    config['defaultmodel'] = model_info['model_type']['tfgpt2'][values['-MODEL-']]
                else:
                    ai = aitextgen(model=model_info['model_type']['nongpt2'][values['-MODEL-']], to_gpu=config['gpubool'], to_fp16=config['use_fp16'], cache_dir=f"./models/{model_info['model_type']['nongpt2'][values['-MODEL-']]}")
                    tokenizer = GPT2Tokenizer.from_pretrained(model_info['model_type']['nongpt2'][values['-MODEL-']])
                    config['defaultmodel'] = model_info['model_type']['nongpt2'][values['-MODEL-']]
                config['nomodel'] = False
                config['model_type'] = model_info['model_type'][values['-MODEL-']]
                break
        if event == '-EXIT-' and (values['-MODEL-'] == 'No model' or not values['-MODEL-']):
            if sg.popup_yes_no('Are you sure you want to use GPT Fewshot Batcher with no model selected and downloaded? You won\'t be able to generate or tokenize anything!', title="Confirm No Model", keep_on_top = True, modal=True) == 'Yes':
                config['nomodel'] = True
                break
    window.close()
    return config, ai, tokenizer

def main():
    # Get No Model mode working again
    if not sg.user_settings_file_exists(filename='config.ini', path='.'):
        # Wonder if it'd be possible to nest all of these?
        config = initialize_config()
        config, ai, tokenizer = first_boot(config)
        # Solve the return issue for No Model mode by separating out the model initialization into a separate function that is also called upon on boots that are not first_boot
        main_window(config, ai, tokenizer)
        main_window(config, ai, tokenizer = first_boot(config = initialize_config))
    else:
        config = sg.UserSettings(filename='config.ini', path='.')
        if config['model_type'] == 'tf_gpt2':
            ai = aitextgen(tf_gpt2=config['defaultmodel'], to_gpu=config['gpubool'], to_fp16=config['use_fp16'], cache_dir=f"./models/gpt2-{config['defaultmodel']}")
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        else:
            ai = aitextgen(model=config['defaultmodel'], to_gpu=config['gpubool'], to_fp16=config['use_fp16'], cache_dir=f"./models/{config['defaultmodel']}")
            tokenizer = GPT2Tokenizer.from_pretrained(config['defaultmodel'])
        main_window(config, ai, tokenizer)

if __name__ == "__main__":
    main()