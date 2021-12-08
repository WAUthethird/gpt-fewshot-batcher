import PySimpleGUI as sg

data = [["Activated", "Test"], ["User-Activated", "Test"], ["Currently Editing", "Test"], ["Deactivated", "Test"], ["Deactivated by contextual generation", "Test"], ["Will be removed if fewshot saved", "Test"]] #The Deactivated by contextual generation color should go away after a pair is saved - not if another generation happens
headings = ["Name", "Test"]
colors = ((0, 'green3'), (1, 'darkorchid1'), (2, 'red'), (3, 'gray26'), (4, 'gray40'), (5, 'darkred'))
othercolors = ((0, 'green'), (1, 'purple'), (2, 'red'), (3, 'grey'), (5, 'darkred'))

def main():
    layout = [[sg.Table(values=data, headings=headings, max_col_width=100,
                                    auto_size_columns=True,
                                    justification='center',
                                    num_rows=min(len(data), 1000),
                                    key='-TABLE-',
                                    expand_x=True,
                                    row_height=100,
                                    row_colors=colors,
                                    enable_events=True,
                                    select_mode=sg.TABLE_SELECT_MODE_BROWSE)],
               [sg.Button('press me for gaming', key='gamingbutton')]]
    window = sg.Window("Table", layout, modal=True, size=(500,600))
    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED:
            break
        if event == '-TABLE-':
            indexprint = values[event]
            valueprint = [data[row] for row in values[event]]
            print(indexprint)
            print(valueprint)
        if event == 'gamingbutton':
            window['-TABLE-'].update(row_colors=othercolors)
    window.close()

if __name__ == '__main__':
    main()