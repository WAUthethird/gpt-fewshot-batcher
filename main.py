import PySimpleGUI as sg

from aitextgen import aitextgen
from transformers import GPT2Tokenizer
import torch
import cpuinfo
from psutil import virtual_memory

sg.theme('Dark Blue 3')


def initialize_config():
    config = sg.UserSettings(filename='config.json', path='.')
    config['automatic_activation'] = True
    config['use_fp16'] = False
    config['gpubool'] = None
    config['nomodel'] = None
    config['defaultmodel'] = None
    config['model_type'] = None
    config['model_context'] = 2048
    config['model_length'] = 128
    config['model_temp'] = 0.9
    config['model_rep_pen'] = 1.0
    config['model_length_pen'] = 1.0
    config['model_top_k'] = 50
    config['model_top_p'] = 1.0
    config['model_fewshotprefix'] = ''
    config['model_after_fewshotprefix'] = ''
    config['model_inputprefix'] = 'Input:'
    config['model_outputprefix'] = 'Output:'
    config['model_after_inputprefix'] = '\n\n'
    config['model_after_inputtext'] = '\n\n'
    config['model_after_outputprefix'] = '\n\n'
    config['model_after_outputtext'] = '\n\n'
    config['model_stopsequence_trim'] = 'model_after_outputprefix'
    config['model_stopsequence'] = '\n\nInput:'
    return config


model_info = {'GPT-Neo 125M': 2, 'GPT-Neo 1.3B': 8, 'GPT-Neo 2.7B': 12, 'GPT-2 124M': 1, 'GPT-2 355M': 4, 'GPT-2 774M': 6, 'GPT-2 1558M': 10, 'model_type': {'GPT-Neo 125M': 'non_gpt2', 'GPT-Neo 1.3B': 'non_gpt2', 'GPT-Neo 2.7B': 'non_gpt2', 'GPT-2 124M': 'tf_gpt2', 'GPT-2 355M': 'tf_gpt2', 'GPT-2 774M': 'tf_gpt2', 'GPT-2 1558M': 'tf_gpt2', 'nongpt2': {'GPT-Neo 125M': 'EleutherAI/gpt-neo-125M', 'GPT-Neo 1.3B': 'EleutherAI/gpt-neo-1.3B', 'GPT-Neo 2.7B': 'EleutherAI/gpt-neo-2.7B'}, 'tfgpt2': {'GPT-2 124M': '124M', 'GPT-2 355M': '355M', 'GPT-2 774M': '774M', 'GPT-2 1558M': '1558M'}}}
colors = {'Activated': 'green3', 'Permanently Activated': 'darkorchid1', 'Editing': 'red', 'Deactivated': 'gray26'}


def main_window(config, ai, tokenizer):
    menu_def = [['&File', ['COULD DO MORE STUFF HERE', '&Settings', 'E&xit']],
                ['&Help', ['&About']]]

    tabledisplay = [['', '', '', '', ''], ['', '', '', '', ''], ['', '', '', '', ''], ['', '', '', '', ''], ['', '', '', '', ''], ['', '', '', '', '']]
    tabledata = []
    assembled_context = ''
    trim_dict = {'After input prefix': 'model_after_inputprefix', 'After input text': 'model_after_inputtext', 'After output prefix': 'model_after_outputprefix', 'After output text': 'model_after_outputtext'}
    headings = ["Row", "          Input          ", "          Output          ", "Token Count", "     Status     "]

    def settings_window():
        text_padding = ((0, 0), (16, 15))
        settings_text = [[sg.Text('Model Context:', pad=text_padding)],
                         [sg.Text('Length:', pad=text_padding)],
                         [sg.Text('Temperature:', pad=text_padding)],
                         [sg.Text('Repetition Penalty:', pad=text_padding)],
                         [sg.Text('Length Penalty:', pad=text_padding)],
                         [sg.Text('Top K:', pad=text_padding)],
                         [sg.Text('Top P:', pad=text_padding)]]

        settings_sliders = [[sg.Slider(range=(1024, 2048), default_value=config['model_context'], key='-MODELCONTEXT-', size=(70, 25), orientation='horizontal')],
                            [sg.Slider(range=(20, 500), default_value=config['model_length'], key='-MODELLENGTH-', size=(70, 25), orientation='horizontal')],
                            [sg.Slider(range=(0.1, 3.0), resolution=0.1, default_value=config['model_temp'], key='-MODELTEMP-', size=(70, 25), orientation='horizontal')],
                            [sg.Slider(range=(0.1, 5.0), resolution=0.1, default_value=config['model_rep_pen'], key='-MODELREP-PEN-', size=(70, 25), orientation='horizontal')],
                            [sg.Slider(range=(0.1, 5.0), resolution=0.1, default_value=config['model_length_pen'], key='-MODELLENGTH-PEN-', size=(70, 25), orientation='horizontal')],
                            [sg.Slider(range=(1, 100), default_value=config['model_top_k'], key='-MODELTOP-K-', size=(70, 25), orientation='horizontal')],
                            [sg.Slider(range=(0.1, 1.0), resolution=0.1, default_value=config['model_top_p'], key='-MODELTOP-P-', size=(70, 25), orientation='horizontal')]]
        # ADD MODEL SELECTION
        # DO IT
        settings_window_main = [[sg.Col([[sg.Text('Settings')]], justification='center')],
                                [sg.Col(settings_text, element_justification='left'), sg.Col(settings_sliders, element_justification='right')],
                                [sg.Text('_'*105)],
                                [sg.Text('Both fewshot prefix fields may be left empty if desired.')],
                                [sg.Text('Fewshot prefix:'), sg.InputText(default_text=config['model_fewshotprefix'].replace('\n', '\\n'), key='-FEWSHOTPREFIX-', size=25, pad=((27, 0), (0, 0)))],
                                [sg.Col([[sg.Text('After fewshot prefix:'), sg.InputText(default_text=config['model_after_fewshotprefix'].replace('\n', '\\n'), key='-AFTER-FEWSHOTPREFIX-', size=25)]], pad=((0, 0), (0, 10)))],
                                [sg.Text('Input prefix:'), sg.InputText(default_text=config['model_inputprefix'].replace('\n', '\\n'), key='-INPUTPREFIX-', size=25, pad=((43, 0), (0, 0)))],
                                [sg.Text('Output prefix:'), sg.InputText(default_text=config['model_outputprefix'].replace('\n', '\\n'), key='-OUTPUTPREFIX-', size=25, pad=((35, 0), (0, 0)))],
                                [sg.Text('After input prefix:'), sg.InputText(default_text=config['model_after_inputprefix'].replace('\n', '\\n'), key='-AFTER-INPUTPREFIX-', size=25, pad=((18, 0), (0, 0)))],
                                [sg.Text('After input text:'), sg.InputText(default_text=config['model_after_inputtext'].replace('\n', '\\n'), key='-AFTER-INPUTTEXT-', size=25, pad=((26, 0), (0, 0)))],
                                [sg.Text('After output prefix:'), sg.InputText(default_text=config['model_after_outputprefix'].replace('\n', '\\n'), key='-AFTER-OUTPUTPREFIX-', size=25, pad=((12, 0), (0, 0)))],
                                [sg.Col([[sg.Text('After output text:'), sg.InputText(default_text=config['model_after_outputtext'].replace('\n', '\\n'), key='-AFTER-OUTPUTTEXT-', size=25, pad=((20, 0), (0, 0)))]], pad=((0, 0), (0, 10)))],
                                [sg.Text('Stop sequence trim:'), sg.Combo(['After input prefix', 'After input text', 'After output prefix', 'After output text'], default_value=[key for key, value in trim_dict.items() if value == config['model_stopsequence_trim']][0], key='-STOPSEQUENCE-TRIM-', size=23, pad=((4, 0), (0, 0)))],
                                [sg.Col([[sg.Text('Stop sequence:'), sg.InputText(default_text=config['model_stopsequence'].replace('\n', '\\n'), key='-STOPSEQUENCE-', size=25, pad=((25, 0), (0, 0)))]], pad=((0, 0), (0, 10)))],
                                [sg.Checkbox('Automatic fewshot activation', default=config['automatic_activation'], key='-AUTOMATIC-ACTIVATION-')],
                                [sg.Col([[sg.Button('Reset to defaults', key='-RESETDEFAULTS-'), sg.Button('Save', key='-SAVESETTINGS-'), sg.Button('Exit', key='-EXITSETTINGS-')]], justification='center')]]

        return sg.Window('Settings', settings_window_main, location=(0, 0), modal=True)

    def pair_display_window(pair):
        pair_display_window_main = [[sg.Multiline(pair, size=(100, 20), key='-PAIRDISPLAYBOX-', disabled=True)],
                                    [sg.Checkbox('Newline Character Display', default=False, key='-NEWLINECHAR-', enable_events=True)]]

        return sg.Window('Pair Display', pair_display_window_main, location=(0, 0))

    side_buttons_table = [[sg.Text('Fewshot List Options')],
                          [sg.Button('Activate pair', key='-ACTIVATEPAIR-')],
                          [sg.Button('Permanently activate pair', key='-PERMACTIVATEPAIR-')],
                          [sg.Button('Deactivate pair', key='-DEACTIVATEPAIR-')],
                          [sg.Button('Display selected pair', key='-DISPLAYPAIR-')],
                          [sg.Button('Edit selected pair', key='-EDITPAIR-')],
                          [sg.Button('Remove selected pair', key='-REMOVEPAIR-')],
                          [sg.Button('Save fewshots to file', key='-SAVEFEWSHOTS-')], # Add detection logic for anything currently being edited and refuse to do it until editing is complete
                          [sg.Button('Load fewshots from file', key='-LOADFEWSHOTS-')], # Add detection logic for anything currently being edited and refuse to do it until editing is complete
                          [sg.Button('Clear fewshot table', key='-CLEARFEWSHOTS-')]] # Add detection logic for anything currently being edited and refuse to do it (or clear everything except for the edited fewshot, displaying a warning message beforehand about this)
    side_buttons_input = [[sg.Text('Current Pair Options', key='-CURRENTPAIRTEXT-')],
                          [sg.Button('Save pair to table', key='-SAVEPAIR-'), sg.Button('Save edits', visible=False, key='-SAVEEDITS-'), sg.Button('Discard edits', visible=False, key='-DISCARDEDITS-')],
                          [sg.Button('(Re)generate output', key='-GENERATE-')],
                          [sg.Button('Clear input and output', key='-CLEAR-')],
                          [sg.Button('Testing button!', key='-TESTING-')]]

    main_layout = [[sg.Menu(menu_def)],
                   [sg.Button('Export'), sg.Button('Settings', key='-SETTINGS-')],
                   [sg.Text(f"Tokens used: 0/{int(config['model_context'])}", key='-TOKENTEXT-')],
                   [sg.Table(values=tabledisplay, headings=headings, max_col_width=100,
                                    background_color='darkblue',
                                    auto_size_columns=True,
                                    justification='center',
                                    num_rows=min(len(tabledisplay), 1000),
                                    key='-TABLE-',
                                    expand_x=True,
                                    row_height=100), sg.Col(side_buttons_table, justification='right', vertical_alignment='top')],
                   [sg.Text('Input', key='-INPUTTEXT-')],
                   [sg.Multiline('', size=(100, 10), key='-INPUTBOX-'), sg.Col(side_buttons_input, justification='right', vertical_alignment='top')],
                   [sg.Text('Output', key='-OUTPUTTEXT-')],
                   [sg.Multiline('', size=(100, 10), key='-OUTPUTBOX-')]]

    window = sg.Window('GPT Fewshot Batcher', main_layout, location=(0, 0))

    while True:
        event, values = window.read()
        # must do tabledisplay = update_table() in all uses

        def update_table():
            if tabledata == []:
                tabledisplay = [['', '', '', '', ''], ['', '', '', '', ''], ['', '', '', '', ''], ['', '', '', '', ''], ['', '', '', '', ''], ['', '', '', '', '']]
                tablecolors = [(0, 'darkblue'), (1, 'darkblue'), (2, 'darkblue'), (3, 'darkblue'), (4, 'darkblue'), (5, 'darkblue'), ]
            else:
                tabledisplay = [[tabledata_index + 1, x['input'], x['output'], x['tokens'], x['status']] for tabledata_index, x in enumerate(tabledata)]
                tablecolors = [((index, colors[x['status']])) for index, x in enumerate(tabledata)]
            window['-TABLE-'].update(values=tabledisplay)
            window['-TABLE-'].update(row_colors=tablecolors)
            return tabledisplay

        def update_token_text():
            tokencount = [x['tokens'] for x in tabledata if x['status'] == 'Activated' or x['status'] == 'Permanently Activated']
            window['-TOKENTEXT-'].update(f"Tokens used: {sum(tokencount)}/{int(config['model_context'])}")

        def assemble_context(assembled_context):
            first_index = True
            for index, value in enumerate(tabledata):
                if (index == 0 or first_index is True) and not value['status'] == 'Deactivated':
                    assembled = f"{config['model_fewshotprefix']}{config['model_after_fewshotprefix']}{config['model_inputprefix']}{config['model_after_inputprefix']}{value['input']}{config['model_after_inputtext']}{config['model_outputprefix']}{config['model_after_outputprefix']}{value['output']}"
                    first_index = False
                elif not value['status'] == 'Deactivated':
                    assembled = f"{config['model_after_outputtext']}{config['model_inputprefix']}{config['model_after_inputprefix']}{value['input']}{config['model_after_inputtext']}{config['model_outputprefix']}{config['model_after_outputprefix']}{value['output']}"
                else:
                    assembled = ''
                assembled_context = f"{assembled_context}{assembled}"
            return assembled_context

        def tokenize_single_fewshot(input_only):
            if tabledisplay[0] == ['', '', '', '', ''] or len(tabledisplay) == 0:
                if input_only:
                    assembled = f"{config['model_fewshotprefix']}{config['model_after_fewshotprefix']}{config['model_after_outputtext']}{config['model_inputprefix']}{config['model_after_inputprefix']}{values['-INPUTBOX-']}{config['model_after_inputtext']}{config['model_outputprefix']}"
                else:
                    assembled = f"{config['model_fewshotprefix']}{config['model_after_fewshotprefix']}{config['model_inputprefix']}{config['model_after_inputprefix']}{values['-INPUTBOX-']}{config['model_after_inputtext']}{config['model_outputprefix']}{config['model_after_outputprefix']}{values['-OUTPUTBOX-']}"
            else:
                if input_only:
                    assembled = f"{config['model_after_outputtext']}{config['model_inputprefix']}{config['model_after_inputprefix']}{values['-INPUTBOX-']}{config['model_after_inputtext']}{config['model_outputprefix']}{config['model_after_outputprefix']}"
                else:
                    assembled = f"{config['model_after_outputtext']}{config['model_inputprefix']}{config['model_after_inputprefix']}{values['-INPUTBOX-']}{config['model_after_inputtext']}{config['model_outputprefix']}{config['model_after_outputprefix']}{values['-OUTPUTBOX-']}"
            assembled_tokens = tokenizer.encode(assembled)
            return assembled_tokens

        def index_for_deactivation():
            indexvalidation = False
            for index, value in enumerate(tabledata):
                if (index == 0 and (value['status'] == 'Deactivated' or value['status'] == 'Permanently Activated')) or indexvalidation is True:
                    indexvalidation = True
                    if value['status'] == 'Activated':
                        indexvalidation = False
                        referenceindex = index
                elif index == 0 and value['status'] == 'Activated':
                    referenceindex = index
            return referenceindex

        def total_token_count():
            token_count = 0
            for value in tabledata:
                if value['status'] == 'Activated' or value['status'] == 'Permanently Activated':
                    temp_token_count = value['tokens']
                    token_count = token_count + temp_token_count
            return token_count

        def generate_text(assemble, assembled_context):
            if assemble:
                tokenized_context = tokenizer.encode(assemble_context(assembled_context))
                tokenized_prompt = tokenize_single_fewshot(True)
                while len(tokenized_prompt) + len(tokenized_context) > (config['model_context'] - config['model_length']):
                    referenceindex = index_for_deactivation()
                    tabledata[referenceindex]['status'] = 'Deactivated'
                    tokenized_context = tokenizer.encode(assemble_context(assembled_context))
                    tokenized_prompt = tokenize_single_fewshot(True)
                assembled_context = assemble_context(assembled_context)
                prompt_temp = f"{assembled_context}{config['model_after_outputtext']}{config['model_inputprefix']}{config['model_after_inputprefix']}{values['-INPUTBOX-']}{config['model_after_inputtext']}{config['model_outputprefix']}"
            else:
                prompt_temp = f"{config['model_fewshotprefix']}{config['model_after_fewshotprefix']}{config['model_inputprefix']}{config['model_after_inputprefix']}{values['-INPUTBOX-']}{config['model_after_inputtext']}{config['model_outputprefix']}"
            prompt_tokens = tokenizer.encode(prompt_temp)
            maxlen = config['model_length'] + len(prompt_tokens)
            gen_text = ai.generate_one(prompt=prompt_temp,
                                       min_length=len(prompt_tokens)+1,
                                       max_length=maxlen,
                                       temperature=config['model_temp'],
                                       repetition_penalty=config['model_rep_pen'],
                                       length_penalty=config['model_length_pen'],
                                       top_k=int(config['model_top_k']),
                                       top_p=config['model_top_p'])
            try:
                gen_stripped_text = gen_text[len(prompt_temp)+len(config[config['model_stopsequence_trim']]):].split(config['model_stopsequence'], 1)[0]
            except:
                gen_stripped_text = gen_text[len(prompt_temp)+len(config[config['model_stopsequence_trim']]):]
            window['-OUTPUTBOX-'].update(gen_stripped_text)
            if assemble:
                tabledisplay = update_table()
                update_token_text()
            else:
                tabledisplay = [['', '', '', '', ''], ['', '', '', '', ''], ['', '', '', '', ''], ['', '', '', '', ''], ['', '', '', '', ''], ['', '', '', '', '']]
            return tabledisplay

        def tokenize_all_fewshots():
            for index, value in enumerate(tabledata):
                if index == 0:
                    assembled = f"{config['model_fewshotprefix']}{config['model_after_fewshotprefix']}{config['model_inputprefix']}{config['model_after_inputprefix']}{value['input']}{config['model_after_inputtext']}{config['model_outputprefix']}{config['model_after_outputprefix']}{value['output']}"
                else:
                    assembled = f"{config['model_after_outputtext']}{config['model_inputprefix']}{config['model_after_inputprefix']}{value['input']}{config['model_after_inputtext']}{config['model_outputprefix']}{config['model_after_outputprefix']}{value['output']}"
                assembled_tokens = tokenizer.encode(assembled)
                value['tokens'] = len(assembled_tokens)

        if event == sg.WIN_CLOSED:
            break
        if event == '-SETTINGS-':
            settings = settings_window()
            while True:
                event, values = settings.read()
                if event == sg.WIN_CLOSED:
                    break
                if event == '-SAVESETTINGS-':
                    temp_settingsdict = {'model_context': config['model_context'], 'model_length': config['model_length'], 'model_fewshotprefix': config['model_fewshotprefix'], 'model_after_fewshotprefix': config['model_after_fewshotprefix'], 'model_inputprefix': config['model_inputprefix'], 'model_outputprefix': config['model_outputprefix'], 'model_after_inputprefix': config['model_after_inputprefix'], 'model_after_inputtext': config['model_after_inputtext'], 'model_after_outputprefix': config['model_after_outputprefix'], 'model_after_outputtext': config['model_after_outputtext']}
                    config['automatic_activation'] = values['-AUTOMATIC-ACTIVATION-']
                    config['model_context'] = values['-MODELCONTEXT-']
                    config['model_length'] = values['-MODELLENGTH-']
                    config['model_temp'] = values['-MODELTEMP-']
                    config['model_rep_pen'] = values['-MODELREP-PEN-']
                    config['model_length_pen'] = values['-MODELLENGTH-PEN-']
                    config['model_top_k'] = values['-MODELTOP-K-']
                    config['model_top_p'] = values['-MODELTOP-P-']
                    config['model_fewshotprefix'] = values['-FEWSHOTPREFIX-'].replace('\\n', '\n')
                    config['model_after_fewshotprefix'] = values['-AFTER-FEWSHOTPREFIX-'].replace('\\n', '\n')
                    config['model_inputprefix'] = values['-INPUTPREFIX-'].replace('\\n', '\n')
                    config['model_outputprefix'] = values['-OUTPUTPREFIX-'].replace('\\n', '\n')
                    config['model_after_inputprefix'] = values['-AFTER-INPUTPREFIX-'].replace('\\n', '\n')
                    config['model_after_inputtext'] = values['-AFTER-INPUTTEXT-'].replace('\\n', '\n')
                    config['model_after_outputprefix'] = values['-AFTER-OUTPUTPREFIX-'].replace('\\n', '\n')
                    config['model_after_outputtext'] = values['-AFTER-OUTPUTTEXT-'].replace('\\n', '\n')
                    config['model_stopsequence_trim'] = trim_dict[values['-STOPSEQUENCE-TRIM-']]
                    config['model_stopsequence'] = values['-STOPSEQUENCE-'].replace('\\n', '\n')
                    if not tabledisplay[0] == ['', '', '', '', ''] and not len(tabledisplay) == 0 and not config['nomodel'] is True:
                        tokencount_permactivated = [x['tokens'] for x in tabledata if x['status'] == 'Permanently Activated']
                        if sum(tokencount_permactivated) > (config['model_context'] - config['model_length']):
                            sg.popup_ok('Token sum of permanently activated fewshots exceeds model context!', title='Error')
                            config['model_context'] = temp_settingsdict['model_context']
                            config['model_length'] = temp_settingsdict['model_length']
                            config['model_fewshotprefix'] = temp_settingsdict['model_fewshotprefix'].replace('\\n', '\n')
                            config['model_after_fewshotprefix'] = temp_settingsdict['model_after_fewshotprefix'].replace('\\n', '\n')
                            config['model_inputprefix'] = temp_settingsdict['model_inputprefix'].replace('\\n', '\n')
                            config['model_outputprefix'] = temp_settingsdict['model_outputprefix'].replace('\\n', '\n')
                            config['model_after_inputprefix'] = temp_settingsdict['model_after_inputprefix'].replace('\\n', '\n')
                            config['model_after_inputtext'] = temp_settingsdict['model_after_inputtext'].replace('\\n', '\n')
                            config['model_after_outputprefix'] = temp_settingsdict['model_after_outputprefix'].replace('\\n', '\n')
                            config['model_after_outputtext'] = temp_settingsdict['model_after_outputtext'].replace('\\n', '\n')
                        tokenize_all_fewshots()
                        token_count_temp = total_token_count()
                        while token_count_temp > (config['model_context'] - config['model_length']):
                            referenceindex = index_for_deactivation()
                            tabledata[referenceindex]['status'] = 'Deactivated'
                            token_count_temp = total_token_count()
                        tabledisplay = update_table()
                    update_token_text()
                    settings['-MODELCONTEXT-'].update(config['model_context'])
                    settings['-MODELLENGTH-'].update(config['model_length'])
                    settings['-FEWSHOTPREFIX-'].update(config['model_fewshotprefix'])
                    settings['-AFTER-FEWSHOTPREFIX-'].update(config['model_after_fewshotprefix'])
                    settings['-INPUTPREFIX-'].update(config['model_inputprefix'])
                    settings['-OUTPUTPREFIX-'].update(config['model_outputprefix'])
                    settings['-AFTER-INPUTPREFIX-'].update(config['model_after_inputprefix'].replace('\n', '\\n'))
                    settings['-AFTER-INPUTTEXT-'].update(config['model_after_inputtext'].replace('\n', '\\n'))
                    settings['-AFTER-OUTPUTPREFIX-'].update(config['model_after_outputprefix'].replace('\n', '\\n'))
                    settings['-AFTER-OUTPUTTEXT-'].update(config['model_after_outputtext'].replace('\n', '\\n'))
                if event == '-RESETDEFAULTS-':
                    if sg.popup_yes_no('Are you sure you want to reset all settings to defaults?', title="Confirm Reset", keep_on_top=True, modal=True) == 'Yes':
                        temp_settingsdict = {'model_context': config['model_context'], 'model_length': config['model_length'], 'model_fewshotprefix': config['model_fewshotprefix'], 'model_after_fewshotprefix': config['model_after_fewshotprefix'], 'model_inputprefix': config['model_inputprefix'], 'model_outputprefix': config['model_outputprefix'], 'model_after_inputprefix': config['model_after_inputprefix'], 'model_after_inputtext': config['model_after_inputtext'], 'model_after_outputprefix': config['model_after_outputprefix'], 'model_after_outputtext': config['model_after_outputtext']}
                        # Replace this with a call to initialize_config() once model switching is up and running
                        config['automatic_activation'] = True
                        config['model_context'] = 2048
                        config['model_length'] = 128
                        config['model_temp'] = 0.9
                        config['model_rep_pen'] = 1.0
                        config['model_length_pen'] = 1.0
                        config['model_top_k'] = 50
                        config['model_top_p'] = 1.0
                        config['model_fewshotprefix'] = ''
                        config['model_after_fewshotprefix'] = ''
                        config['model_inputprefix'] = 'Input:'
                        config['model_outputprefix'] = 'Output:'
                        config['model_after_inputprefix'] = '\n\n'
                        config['model_after_inputtext'] = '\n\n'
                        config['model_after_outputprefix'] = '\n\n'
                        config['model_after_outputtext'] = '\n\n'
                        config['model_stopsequence_trim'] = 'model_after_outputprefix'
                        config['model_stopsequence'] = '\n\nInput:'
                        if not tabledisplay[0] == ['', '', '', '', ''] and not len(tabledisplay) == 0 and not config['nomodel'] is True:
                            tokencount_permactivated = [x['tokens'] for x in tabledata if x['status'] == 'Permanently Activated']
                            if sum(tokencount_permactivated) > (config['model_context'] - config['model_length']):
                                sg.popup_ok('Token sum of permanently activated fewshots exceeds model context!', title='Error')
                                config['model_context'] = temp_settingsdict['model_context']
                                config['model_length'] = temp_settingsdict['model_length']
                                config['model_fewshotprefix'] = temp_settingsdict['model_fewshotprefix'].replace('\\n', '\n')
                                config['model_after_fewshotprefix'] = temp_settingsdict['model_after_fewshotprefix'].replace('\\n', '\n')
                                config['model_inputprefix'] = temp_settingsdict['model_inputprefix'].replace('\\n', '\n')
                                config['model_outputprefix'] = temp_settingsdict['model_outputprefix'].replace('\\n', '\n')
                                config['model_after_inputprefix'] = temp_settingsdict['model_after_inputprefix'].replace('\\n', '\n')
                                config['model_after_inputtext'] = temp_settingsdict['model_after_inputtext'].replace('\\n', '\n')
                                config['model_after_outputprefix'] = temp_settingsdict['model_after_outputprefix'].replace('\\n', '\n')
                                config['model_after_outputtext'] = temp_settingsdict['model_after_outputtext'].replace('\\n', '\n')
                            tokenize_all_fewshots()
                            token_count_temp = total_token_count()
                            while token_count_temp > (config['model_context'] - config['model_length']):
                                referenceindex = index_for_deactivation()
                                tabledata[referenceindex]['status'] = 'Deactivated'
                                token_count_temp = total_token_count()
                            tabledisplay = update_table()
                        update_token_text()
                        settings['-AUTOMATIC-ACTIVATION-'].update(config['automatic_activation'])
                        settings['-MODELCONTEXT-'].update(config['model_context'])
                        settings['-MODELLENGTH-'].update(config['model_length'])
                        settings['-MODELTEMP-'].update(config['model_temp'])
                        settings['-MODELREP-PEN-'].update(config['model_rep_pen'])
                        settings['-MODELLENGTH-PEN-'].update(config['model_length_pen'])
                        settings['-MODELTOP-K-'].update(config['model_top_k'])
                        settings['-MODELTOP-P-'].update(config['model_top_p'])
                        settings['-FEWSHOTPREFIX-'].update(config['model_fewshotprefix'])
                        settings['-AFTER-FEWSHOTPREFIX-'].update(config['model_after_fewshotprefix'])
                        settings['-INPUTPREFIX-'].update(config['model_inputprefix'])
                        settings['-OUTPUTPREFIX-'].update(config['model_outputprefix'])
                        settings['-AFTER-INPUTPREFIX-'].update(config['model_after_inputprefix'].replace('\n', '\\n'))
                        settings['-AFTER-INPUTTEXT-'].update(config['model_after_inputtext'].replace('\n', '\\n'))
                        settings['-AFTER-OUTPUTPREFIX-'].update(config['model_after_outputprefix'].replace('\n', '\\n'))
                        settings['-AFTER-OUTPUTTEXT-'].update(config['model_after_outputtext'].replace('\n', '\\n'))
                        settings['-STOPSEQUENCE-TRIM-'].update([key for key, value in trim_dict.items() if value == config['model_stopsequence_trim']][0])
                        settings['-STOPSEQUENCE-'].update(config['model_stopsequence'].replace('\n', '\\n'))
                if event == '-EXITSETTINGS-':
                    break
            settings.close()
        if event == '-GENERATE-':
            if config['nomodel'] is True:
                sg.popup_ok('You must have a model loaded to generate text!', title='Error')
            else:
                if values['-INPUTBOX-'] == '':
                    sg.popup_ok('Input box must have text!', title='Error')
                else:
                    if tabledisplay[0] == ['', '', '', '', ''] or len(tabledisplay) == 0:
                        if sg.popup_yes_no('Are you sure you\'d like to generate text with an empty context?', title='Confirm generation') == 'Yes':
                            assemble = False
                            tabledisplay = generate_text(assemble, assembled_context)
                    else:
                        assemble = True
                        tabledisplay = generate_text(assemble, assembled_context)
        if event == '-SAVEPAIR-':
            if values['-INPUTBOX-'] == '' or values['-OUTPUTBOX-'] == '':
                sg.popup_ok('Both text boxes must have text!', title='Error')
            else:
                if not config['nomodel'] is True:
                    assembled_tokens = tokenize_single_fewshot(False)
                    token_count = total_token_count()
                    if len(assembled_tokens) > (config['model_context'] - config['model_length']):
                        sg.popup_ok(f"Your fewshot pair exceeds the maximum allowed length ({config['model_context'] - config['model_length']})! Please lower the length and try again.", title='Error')
                    else:
                        tokencount_permactivated = [x['tokens'] for x in tabledata if x['status'] == 'Permanently Activated']
                        if (sum(tokencount_permactivated) + len(assembled_tokens)) > (config['model_context'] - config['model_length']) or config['automatic_activation'] is False:
                            save_activated = False
                        else:
                            save_activated = True
                            # THIS SHOULD BE CHECKING FOR WHAT MODE YOU ARE ON AS WELL
                            while token_count + len(assembled_tokens) > (config['model_context'] - config['model_length']):
                                referenceindex = index_for_deactivation()
                                tabledata[referenceindex]['status'] = 'Deactivated'
                                token_count = total_token_count()
                else:
                    assembled_tokens = ''
                    if config['automatic_activation']:
                        save_activated = True
                    else:
                        save_activated = False
                if not len(assembled_tokens) > (config['model_context'] - config['model_length']):
                    # This should support modes soon
                    if save_activated:
                        tempdict = {'input': values['-INPUTBOX-'], 'output': values['-OUTPUTBOX-'], 'tokens': len(assembled_tokens), 'status': 'Activated'}
                    else:
                        tempdict = {'input': values['-INPUTBOX-'], 'output': values['-OUTPUTBOX-'], 'tokens': len(assembled_tokens), 'status': 'Deactivated'}
                    tabledata.append(tempdict)
                    tabledisplay = update_table()
                    update_token_text()
                    window['-INPUTBOX-'].update('')
                    window['-OUTPUTBOX-'].update('')
        if event == '-ACTIVATEPAIR-':
            if tabledisplay[0] == ['', '', '', '', ''] or len(tabledisplay) == 0:
                sg.popup_ok('No pairs to activate!', title='Error')
            elif values['-TABLE-'] == []:
                sg.popup_ok('Must select a pair to activate!', title='Error')
            elif len(values['-TABLE-']) > 1:
                # Consider changing this
                sg.popup_ok('Cannot activate more than one pair at a time!', title='Error')
            elif tabledata[values['-TABLE-'][0]]['status'] == 'Editing':
                sg.popup_ok('Cannot activate/change status of editing pair!', title='Error')
            else:
                if not tabledisplay[0] == ['', '', '', '', ''] and not len(tabledisplay) == 0 and not config['nomodel'] is True:
                    tabledata[values['-TABLE-'][0]]['status'] = 'Activated'
                    tokenize_all_fewshots()
                    token_count_temp = total_token_count()
                    while token_count_temp > (config['model_context'] - config['model_length']):
                        referenceindex = index_for_deactivation()
                        tabledata[referenceindex]['status'] = 'Deactivated'
                        token_count_temp = total_token_count()
                    editingindex = dict((v['status'], i) for i, v in enumerate(tabledata)).get('Editing', -1)
                    tabledisplay = update_table()
                update_token_text()
        if event == '-PERMACTIVATEPAIR-':
            if tabledisplay[0] == ['', '', '', '', ''] or len(tabledisplay) == 0:
                sg.popup_ok('No pairs to permanently activate!', title='Error')
            elif values['-TABLE-'] == []:
                sg.popup_ok('Must select a pair to permanently activate!', title='Error')
            elif len(values['-TABLE-']) > 1:
                # Consider changing this
                sg.popup_ok('Cannot permanently activate more than one pair at a time!', title='Error')
            elif tabledata[values['-TABLE-'][0]]['status'] == 'Editing':
                sg.popup_ok('Cannot permanently activate/change status of editing pair!', title='Error')
            else:
                if not tabledisplay[0] == ['', '', '', '', ''] and not len(tabledisplay) == 0 and not config['nomodel'] is True:
                    status_storage = tabledata[values['-TABLE-'][0]]['status']
                    tabledata[values['-TABLE-'][0]]['status'] = 'Permanently Activated'
                    tokenize_all_fewshots()
                    tokencount_permactivated = [x['tokens'] for x in tabledata if x['status'] == 'Permanently Activated']
                    if sum(tokencount_permactivated) > (config['model_context'] - config['model_length']):
                        sg.popup_ok('Token sum of permanently activated fewshots would exceed model context!', title='Error')
                        tabledata[values['-TABLE-'][0]]['status'] = status_storage
                        tokenize_all_fewshots()
                    else:
                        token_count_temp = total_token_count()
                        while token_count_temp > (config['model_context'] - config['model_length']):
                            referenceindex = index_for_deactivation()
                            tabledata[referenceindex]['status'] = 'Deactivated'
                            token_count_temp = total_token_count()
                        editingindex = dict((v['status'], i) for i, v in enumerate(tabledata)).get('Editing', -1)
                        tabledisplay = update_table()
                update_token_text()
        if event == '-DEACTIVATEPAIR-':
            if tabledisplay[0] == ['', '', '', '', ''] or len(tabledisplay) == 0:
                sg.popup_ok('No pairs to deactivate!', title='Error')
            elif values['-TABLE-'] == []:
                sg.popup_ok('Must select a pair to deactivate!', title='Error')
            elif len(values['-TABLE-']) > 1:
                # Consider changing this
                sg.popup_ok('Cannot deactivate more than one pair at a time!', title='Error')
            elif tabledata[values['-TABLE-'][0]]['status'] == 'Editing':
                sg.popup_ok('Cannot deactivate/change status of editing pair!', title='Error')
            else:
                if not tabledisplay[0] == ['', '', '', '', ''] and not len(tabledisplay) == 0 and not config['nomodel'] is True:
                    tabledata[values['-TABLE-'][0]]['status'] = 'Deactivated'
                    tokenize_all_fewshots()
                    editingindex = dict((v['status'], i) for i, v in enumerate(tabledata)).get('Editing', -1)
                    tabledisplay = update_table()
                update_token_text()
        if event == '-DISPLAYPAIR-':
            if tabledisplay[0] == ['', '', '', '', ''] or len(tabledisplay) == 0:
                sg.popup_ok('No pairs to display!', title='Error')
            elif values['-TABLE-'] == []:
                sg.popup_ok('Must select a pair to display!', title='Error')
            elif len(values['-TABLE-']) > 1:
                sg.popup_ok('Cannot display more than one pair at a time!', title='Error')
            else:
                if tabledata[values['-TABLE-'][0]]['status'] == 'Editing':
                    sg.popup_ok('Please note that this will show the non-edited version only, until your edits are saved.', title='Alert')
                if values['-TABLE-'][0] == 0:
                    assembled = f"{config['model_fewshotprefix']}{config['model_after_fewshotprefix']}{config['model_inputprefix']}{config['model_after_inputprefix']}{tabledata[values['-TABLE-'][0]]['input']}{config['model_after_inputtext']}{config['model_outputprefix']}{config['model_after_outputprefix']}{tabledata[values['-TABLE-'][0]]['output']}"
                else:
                    assembled = f"{config['model_after_outputtext']}{config['model_inputprefix']}{config['model_after_inputprefix']}{tabledata[values['-TABLE-'][0]]['input']}{config['model_after_inputtext']}{config['model_outputprefix']}{config['model_after_outputprefix']}{tabledata[values['-TABLE-'][0]]['output']}"
                pairdisplay = pair_display_window(assembled)
                while True:
                    event, values = pairdisplay.read()
                    if event == sg.WIN_CLOSED:
                        break
                    if event == '-NEWLINECHAR-':
                        if values['-NEWLINECHAR-']:
                            assembled = assembled.replace('\n', '\\n')
                        else:
                            assembled = assembled.replace('\\n', '\n')
                        pairdisplay['-PAIRDISPLAYBOX-'].update(assembled)
                pairdisplay.close()
        if event == '-EDITPAIR-':
            if tabledisplay[0] == ['', '', '', '', ''] or len(tabledisplay) == 0:
                sg.popup_ok('No pairs to edit!', title='Error')
            elif values['-TABLE-'] == []:
                sg.popup_ok('Must select a pair to edit!', title='Error')
            elif not next((item for item in tabledata if item['status'] == 'Editing'), None) == None:
                sg.popup_ok('Cannot edit another pair until current editing is complete!', title='Error')
            elif len(values['-TABLE-']) > 1:
                sg.popup_ok('Cannot edit more than one pair at a time!', title='Error')
            else:
                edit = False
                if not values['-INPUTBOX-'] == '' or not values['-OUTPUTBOX-'] == '':
                    if sg.popup_yes_no('This will clear what you currently have in your input/output boxes. Continue?', title='Confirm clear') == 'Yes':
                        edit = True
                else:
                    edit = True
                if edit:
                    status_storage = tabledata[values['-TABLE-'][0]]['status']
                    window['-SAVEPAIR-'].update(visible=False)
                    window['-SAVEEDITS-'].update(visible=True)
                    window['-DISCARDEDITS-'].update(visible=True)
                    tabledata[values['-TABLE-'][0]]['status'] = 'Editing'
                    editingindex = dict((v['status'], i) for i, v in enumerate(tabledata)).get('Editing', -1)
                    window['-INPUTBOX-'].update(tabledata[values['-TABLE-'][0]]['input'])
                    window['-OUTPUTBOX-'].update(tabledata[values['-TABLE-'][0]]['output'])
                    tabledisplay = update_table()
                    update_token_text()
        if event == '-SAVEEDITS-':
            if values['-INPUTBOX-'] == '' or values['-OUTPUTBOX-'] == '':
                sg.popup_ok('Both text boxes must have text!', title='Error')
            else:
                if sg.popup_yes_no('Are you sure?', title='Confirm save') == 'Yes':
                    if not config['nomodel'] is True:
                        quit_save_edit = False
                        assembled_tokens = tokenize_single_fewshot(False)
                        token_count = total_token_count()
                        if len(assembled_tokens) > (config['model_context'] - config['model_length']):
                            sg.popup_ok(f"Your fewshot pair exceeds the maximum allowed length ({config['model_context'] - config['model_length']})! Please lower the length and try again.", title='Error')
                        else:
                            tabledata[editingindex]['status'] = status_storage
                            input_storage = tabledata[editingindex]['input']
                            output_storage = tabledata[editingindex]['output']
                            tabledata[editingindex]['input'] = values['-INPUTBOX-']
                            tabledata[editingindex]['output'] = values['-OUTPUTBOX-']
                            tokenize_all_fewshots()
                            tokencount_permactivated = [x['tokens'] for x in tabledata if x['status'] == 'Permanently Activated']
                            if (sum(tokencount_permactivated) > (config['model_context'] - config['model_length'])) and status_storage == 'Permanently Activated':
                                if sg.popup_yes_no('Token sum of permanently activated fewshots would exceed model context! Would you like to save as deactivated?', title='Error') == 'Yes':
                                    tabledata[editingindex]['status'] = 'Deactivated'
                                    tokenize_all_fewshots()
                                else:
                                    tabledata[editingindex]['status'] = status_storage
                                    tabledata[editingindex]['input'] = input_storage
                                    tabledata[editingindex]['output'] = output_storage
                                    tokenize_all_fewshots()
                                    quit_save_edit = True
                            elif (sum(tokencount_permactivated) > (config['model_context'] - config['model_length'])) and status_storage == 'Activated' or status_storage == 'Deactivated':
                                tabledata[editingindex]['status'] = 'Deactivated'
                                tokenize_all_fewshots()
                            else:
                                token_count_temp = total_token_count()
                                while token_count_temp > (config['model_context'] - config['model_length']):
                                    referenceindex = index_for_deactivation()
                                    tabledata[referenceindex]['status'] = 'Deactivated'
                                    token_count_temp = total_token_count()
                            if not quit_save_edit:
                                window['-INPUTBOX-'].update('')
                                window['-OUTPUTBOX-'].update('')
                                window['-SAVEPAIR-'].update(visible=True)
                                window['-SAVEEDITS-'].update(visible=False)
                                window['-DISCARDEDITS-'].update(visible=False)
                                tabledisplay = update_table()
                                update_token_text()
        if event == '-DISCARDEDITS-':
            if sg.popup_yes_no('Are you sure?', title='Confirm discard') == 'Yes':
                if not config['nomodel'] is True:
                    tabledata[editingindex]['status'] = status_storage
                    window['-INPUTBOX-'].update('')
                    window['-OUTPUTBOX-'].update('')
                    window['-SAVEPAIR-'].update(visible=True)
                    window['-SAVEEDITS-'].update(visible=False)
                    window['-DISCARDEDITS-'].update(visible=False)
                    tabledisplay = update_table()
                    update_token_text()
        if event == '-REMOVEPAIR-':
            if tabledisplay[0] == ['', '', '', '', ''] or len(tabledisplay) == 0:
                sg.popup_ok('No pairs to remove!', title='Error')
            elif values['-TABLE-'] == []:
                sg.popup_ok('Must select a pair to remove!', title='Error')
            elif len(values['-TABLE-']) > 1:
                # Consider changing this
                sg.popup_ok('Cannot remove more than one pair at a time!', title='Error')
            elif tabledata[values['-TABLE-'][0]]['status'] == 'Editing':
                sg.popup_ok('Cannot remove editing pair!', title='Error')
            else:
                if sg.popup_yes_no('Are you sure?', title='Confirm remove') == 'Yes':
                    tabledata.remove(tabledata[values['-TABLE-'][0]])
                    if not config['nomodel'] is True:
                        tokenize_all_fewshots()
                        tabledisplay = update_table()
                        update_token_text()
        if event == '-CLEAR-':
            if sg.popup_yes_no('Are you sure?', title='Confirm clear') == 'Yes':
                window['-INPUTBOX-'].update('')
                window['-OUTPUTBOX-'].update('')
        if event == '-TESTING-':
            print('I don\'t do anything right now!')
    window.close()


def initialize_ai(config):
    if config['model_type'] == 'tf_gpt2':
        ai = aitextgen(tf_gpt2=config['defaultmodel'], to_gpu=config['gpubool'], to_fp16=config['use_fp16'], cache_dir=f"./models/gpt2-{config['defaultmodel']}")
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    else:
        ai = aitextgen(model=config['defaultmodel'], to_gpu=config['gpubool'], to_fp16=config['use_fp16'], cache_dir=f"./models/{config['defaultmodel']}")
        tokenizer = GPT2Tokenizer.from_pretrained(config['defaultmodel'])
    return ai, tokenizer


def first_boot(config):
    if torch.cuda.is_available():
        gpusupport = 'YES'
        showgpustuff = True
        devicename = torch.cuda.get_device_name()
        deviceramtext = 'GPU VRAM: '
        deviceram = str(round(torch.cuda.get_device_properties(0).total_memory / (1024.0 ** 3)))
        devicecolor = 'lightgreen'
        config['gpubool'] = True
    else:
        gpusupport = 'NO'
        showgpustuff = False
        devicename = f"Will run on {str(cpuinfo.get_cpu_info()['brand_raw'])} instead"
        deviceramtext = 'System RAM: '
        deviceram = str(round(virtual_memory().total / (1024.0 ** 3)))
        devicecolor = 'orange'
        config['gpubool'] = False

    layout = [[sg.Text("First boot detected! It is recommended that you select and download an AI model before continuing.")],
              [sg.Text("GPU detected:"), sg.Text(f"{gpusupport} - {devicename}", text_color=devicecolor, key='-GPUSUPPORTTEXT-')],
              [sg.Text(deviceramtext, key='-DEVICERAMPREFIX-'), sg.Text(f"{deviceram} GB", text_color=devicecolor, key='-DEVICERAMTEXT-')],
              [sg.Checkbox('GPU Enabled', default=showgpustuff, visible=showgpustuff, key='-GPUCHECKBOX-', enable_events=True)],
              [sg.Text("Available models:")],
              [sg.Combo(['No model', 'GPT-Neo 125M', 'GPT-Neo 1.3B', 'GPT-Neo 2.7B', 'GPT-2 124M', 'GPT-2 355M', 'GPT-2 774M', 'GPT-2 1558M'], key='-MODEL-', enable_events=True), sg.Checkbox('FP16', default=config['use_fp16'], visible=False, key='-FP16CHECKBOX-', enable_events=True)],
              [sg.Text("Great! You should be able to run this model!", text_color='lightgreen', visible=False, key='-CANRUNMODEL-'), sg.Text("Uh oh... it looks like you don't meet the minimum memory requirements for this model. You can still try to run it, but it may not work.", text_color='orange', visible=False, key='-CANTRUNMODEL-')],
              [sg.Column([[sg.Button("Select & Download", key='-SELECT-', visible=False), sg.Button("Exit", key='-EXIT-')]], justification='center', vertical_alignment='top')]]
    window = sg.Window("Model Selection", layout, modal=True)
    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED:
            config['nomodel'] = True
            break
        if values['-MODEL-'] and not values['-MODEL-'] == 'No model':
            window['-SELECT-'].update(visible=True)
            if config['gpubool'] is True:
                window['-FP16CHECKBOX-'].update(visible=True)
            if int(deviceram) >= model_info[values['-MODEL-']] or (values['-FP16CHECKBOX-'] is True and int(deviceram) >= model_info[values['-MODEL-']] // 2):
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
        if event == '-GPUCHECKBOX-' and showgpustuff is True:
            if values['-GPUCHECKBOX-'] is False:
                gpusupport = 'NO'
                devicename = f"Will run on {str(cpuinfo.get_cpu_info()['brand_raw'])} instead"
                deviceramtext = 'System RAM: '
                deviceram = str(round(virtual_memory().total / (1024.0 ** 3)))
                devicecolor = 'red'
                config['gpubool'] = False
                config['use_fp16'] = False
                window['-FP16CHECKBOX-'].update(visible=False)
            else:
                gpusupport = 'YES'
                devicename = torch.cuda.get_device_name()
                deviceramtext = 'GPU VRAM: '
                deviceram = str(round(torch.cuda.get_device_properties(0).total_memory / (1024.0 ** 3)))
                devicecolor = 'lightgreen'
                config['gpubool'] = True
                if not values['-MODEL-'] == '' and not values['-MODEL-'] == 'No model':
                    window['-FP16CHECKBOX-'].update(visible=True)
            if values['-MODEL-'] and not values['-MODEL-'] == 'No model':
                if int(deviceram) >= model_info[values['-MODEL-']] or (values['-FP16CHECKBOX-'] is True and int(deviceram) >= model_info[values['-MODEL-']] // 2):
                    window['-CANTRUNMODEL-'].update(visible=False)
                    window['-CANRUNMODEL-'].update(visible=True)
                else:
                    window['-CANTRUNMODEL-'].update(visible=True)
                    window['-CANRUNMODEL-'].update(visible=False)
            else:
                window['-CANTRUNMODEL-'].update(visible=False)
                window['-CANRUNMODEL-'].update(visible=False)
            window['-DEVICERAMPREFIX-'].update(deviceramtext)
            window['-GPUSUPPORTTEXT-'].update(f"{gpusupport} - {devicename}", text_color=devicecolor)
            window['-DEVICERAMTEXT-'].update(f"{deviceram} GB", text_color=devicecolor)
        if event == '-FP16CHECKBOX-':
            config['use_fp16'] = values['-FP16CHECKBOX-']
        if event == '-SELECT-' and not values['-MODEL-'] == 'No model':
            if sg.popup_yes_no(f"Are you sure you want to download the {values['-MODEL-']} model?", title="Confirm Model Selection", keep_on_top=True, modal=True) == 'Yes':
                if model_info['model_type'][values['-MODEL-']] == 'tf_gpt2':
                    config['defaultmodel'] = model_info['model_type']['tfgpt2'][values['-MODEL-']]
                else:
                    config['defaultmodel'] = model_info['model_type']['nongpt2'][values['-MODEL-']]
                config['nomodel'] = False
                config['model_type'] = model_info['model_type'][values['-MODEL-']]
                break
        if event == '-EXIT-' and (values['-MODEL-'] == 'No model' or not values['-MODEL-']):
            if sg.popup_yes_no('Are you sure you want to use GPT Fewshot Batcher with no model selected and downloaded? You won\'t be able to generate or tokenize anything!', title="Confirm No Model", keep_on_top=True, modal=True) == 'Yes':
                config['nomodel'] = True
                break
    window.close()
    return config


def main():
    if not sg.user_settings_file_exists(filename='config.json', path='.'):
        config = initialize_config()
        config = first_boot(config)
    else:
        config = sg.UserSettings(filename='config.json', path='.')
    if config['nomodel'] is True:
        ai = None
        tokenizer = None
    else:
        ai, tokenizer = initialize_ai(config)
    main_window(config, ai, tokenizer)


if __name__ == "__main__":
    main()