import os
import tkinter
import tkinter.font
import tkinter.filedialog
import tkinter.messagebox
from Main import PageRankMain

loadPath = None
savePath = os.getcwd().replace('\\', '/')


def Action_OpenInput():
    global loadPath
    filePath = tkinter.filedialog.askopenfilename(title=u'选择文件')
    labelInputPath.config(text=filePath)
    loadPath = filePath


def Action_OpenOutput():
    global savePath
    filePath = tkinter.filedialog.askdirectory(title=u'选择文件夹', initialdir=(os.path.expanduser(os.getcwd())))
    labelOutputPath.config(text=filePath)
    savePath = filePath


def Action_ExecutionPageRank():
    global loadPath, savePath
    if loadPath is None or savePath is None:
        tkinter.messagebox.showwarning("WARNING", 'Please give a correct input or output path.')
        return
    if checkVarTransport.get() == 1:
        transportRatio = textTransport.get('0.0', 'end').replace('\n', '')
    else:
        transportRatio = -1
    minChange = textMinChange.get('0.0', 'end').replace('\n', '')
    maxIteration = textMaxIteration.get('0.0', 'end').replace('\n', '')
    topN = textTopN.get('0.0', 'end').replace('\n', '')

    command = '--input_path="%s" --output_path="%s"' % (loadPath, savePath)
    command += ' --teleport_parameter=%s' % transportRatio
    if checkVarDeadEnd.get() == 0: command += ' --not_dead_end_flag'
    command += ' --min_changes=%s' % minChange
    command += ' --max_iteration_times=%s' % maxIteration
    command += ' --top_node_number=%s' % topN

    if radioVarDense.get() == 1:
        command += ' --dense_flag'
        if checkVarPowerFlag.get() == 1:
            command += ' --power_flag'
    else:
        if checkVarBlockMatrix.get() == 1:
            command += ' --block_flag'
            blockSize = textBlockMlength.get('0.0', 'end').replace('\n', '')
            command += ' --block_length=%s' % blockSize
    print(command)
    PageRankMain(command=command)


if __name__ == '__main__':
    win = tkinter.Tk()
    win.title('Page Rank Experiment')
    win.geometry('800x280')

    buttonOpenInput = tkinter.Button(
        win, text='Open Input', font=tkinter.font.Font(family='Times New Roman', size=14), command=Action_OpenInput,
        width=12)
    buttonOpenOutput = tkinter.Button(
        win, text='Change Output', font=tkinter.font.Font(family='Times New Roman', size=14), command=Action_OpenOutput,
        width=12)
    labelInputLegend = tkinter.Label(
        win, text='Input Path', font=tkinter.font.Font(family='Times New Roman', size=18))
    labelOutputLegend = tkinter.Label(
        win, text='Output Path', font=tkinter.font.Font(family='Times New Roman', size=18))
    labelInputPath = tkinter.Label(
        win, text='Not Appointed', font=tkinter.font.Font(family='Times New Roman', size=14))
    labelOutputPath = tkinter.Label(
        win, text=os.getcwd().replace('\\', '/') + '/', font=tkinter.font.Font(family='Times New Roman', size=14))

    buttonOpenInput.place(x=10, y=10)
    buttonOpenOutput.place(x=10, y=50)
    labelInputLegend.place(x=150, y=10)
    labelInputPath.place(x=300, y=15)
    labelOutputLegend.place(x=150, y=50)
    labelOutputPath.place(x=300, y=55)

    checkVarTransport = tkinter.IntVar()
    checkVarDeadEnd = tkinter.IntVar()
    checkButtonTransport = tkinter.Checkbutton(
        win, text='Transport Ratio', font=tkinter.font.Font(family='Times New Roman', size=16),
        variable=checkVarTransport, onvalue=1, offvalue=0)
    textTransport = tkinter.Text(win, font=tkinter.font.Font(family='Times New Roman', size=14), height=1, width=8)
    checkButtonDeadEnd = tkinter.Checkbutton(
        win, text='Consider Dead End', font=tkinter.font.Font(family='Times New Roman', size=16),
        variable=checkVarDeadEnd, onvalue=1, offvalue=0)
    labelMinChangeLegend = tkinter.Label(
        win, text='Min Changes', font=tkinter.font.Font(family='Times New Roman', size=16))
    textMinChange = tkinter.Text(win, font=tkinter.font.Font(family='Times New Roman', size=14), height=1, width=8)
    labelMaxIterationLegend = tkinter.Label(
        win, text='Max Iteration', font=tkinter.font.Font(family='Times New Roman', size=16))
    textMaxIteration = tkinter.Text(win, font=tkinter.font.Font(family='Times New Roman', size=14), height=1, width=8)
    labelTopNLegend = tkinter.Label(
        win, text='Top N Node', font=tkinter.font.Font(family='Times New Roman', size=16))
    textTopN = tkinter.Text(win, font=tkinter.font.Font(family='Times New Roman', size=14), height=1, width=8)
    checkButtonTransport.select()
    checkButtonDeadEnd.select()

    textTransport.insert(0.0, '0.85')
    textMinChange.insert(0.0, '1E-3')
    textMaxIteration.insert(0.0, '1000')
    textTopN.insert(0.0, '100')

    checkButtonTransport.place(x=10, y=120)
    textTransport.place(x=220, y=122)
    checkButtonDeadEnd.place(x=10, y=150)
    labelMinChangeLegend.place(x=35, y=180)
    textMinChange.place(x=220, y=182)
    labelMaxIterationLegend.place(x=35, y=210)
    textMaxIteration.place(x=220, y=212)
    labelTopNLegend.place(x=35, y=240)
    textTopN.place(x=220, y=242)

    radioVarDense = tkinter.IntVar()
    checkVarBlockMatrix = tkinter.IntVar()
    checkVarPowerFlag = tkinter.IntVar()
    radioButtonSparse = tkinter.Radiobutton(
        win, text='Sparse Treatment', value=0, variable=radioVarDense,
        font=tkinter.font.Font(family='Times New Roman', size=16))
    radioButtonDense = tkinter.Radiobutton(
        win, text='Dense Treatment', value=1, variable=radioVarDense,
        font=tkinter.font.Font(family='Times New Roman', size=16))
    checkButtonBlock = tkinter.Checkbutton(
        win, text='Block Size', font=tkinter.font.Font(family='Times New Roman', size=16),
        variable=checkVarBlockMatrix, onvalue=1, offvalue=0)
    textBlockMlength = tkinter.Text(win, font=tkinter.font.Font(family='Times New Roman', size=14), height=1, width=8)
    checkButtonPower = tkinter.Checkbutton(
        win, text='Dense Matrix 2^K', font=tkinter.font.Font(family='Times New Roman', size=16),
        variable=checkVarPowerFlag, onvalue=1, offvalue=0)

    textBlockMlength.insert(0.0, '1000')

    radioButtonSparse.select()
    radioButtonSparse.place(x=320, y=120)
    radioButtonDense.place(x=600, y=120)
    checkButtonBlock.place(x=320, y=150)
    checkButtonPower.place(x=600, y=150)
    textBlockMlength.place(x=500, y=155)

    buttonExecution = tkinter.Button(
        win, text='Execution', font=tkinter.font.Font(family='Times New Roman', size=24),
        command=Action_ExecutionPageRank)
    buttonExecution.place(x=490, y=200)

    win.mainloop()
