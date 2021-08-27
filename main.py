import PySimpleGUI as sg

sg.theme('Dark Blue 3')

data = [["INPUT", "OUTPUT", "TOKENCOUNT"], ["INPUT2", "OUTPUT2", "TOKENCOUNT2"], ["INPUT3", "OUTPUT3", "TOKENCOUNT3"], ["INPUT4", "OUTPUT4", "TOKENCOUNT4"], ["INPUT5", "OUTPUT5", "TOKENCOUNT5"], ["INPUT6", "OUTPUT6", "TOKENCOUNT6"], ["INPUT7", "OUTPUT7", "TOKENCOUNT7"], ["INPUT8", "OUTPUT8", "TOKENCOUNT8"], ["Line1\nLine2\nLine3", "Line1\nLine2\nLine3", "Line1\nLine2\nLine3"]]
headings = ["Input", "Output", "Token Count"]

side_buttons_table = [[sg.Text('Fewshot List Options')],
                      [sg.Button('Display selected pair')], # This should generate a popup asking if they're sure - it'll wipe whatever they've already got in there. We don't need to show the popup if they don't have anything in the boxes, though.
                      [sg.Button('Remove selected pair')],
                      [sg.Button('Save fewshots to file')],
                      [sg.Button('Load fewshots from file')],
                      [sg.Button('Clear fewshot table')]]

side_buttons_input = [[sg.Text('Current Pair Options')],
                      [sg.Button('Save pair to table')],
                      [sg.Button('Regen output')],
                      [sg.Button('Preview trimmed pairs')],
                      [sg.Button('Clear input and output')]]

main_layout = [[sg.Button('Change input prefix'), sg.Button('Change output prefix'), sg.Button('Export')],
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

window = sg.Window('GPT Fewshot Batcher', main_layout, location=(0,0))

event, values = window.read()
window.close()