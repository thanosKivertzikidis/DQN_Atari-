import matplotlib.pyplot as plt
def plotTrainingInfo(data1:list,data2:list,data3:list,data4:list,plot_title:str,plotname:str)->None:
    x_values = range(len(data1))
    fig, (plot1, plot2) = plt.subplots(1, 2, figsize=(10, 5))


    plot1.plot(x_values, data1, label='Score of each episode')
    plot1.plot(x_values, data2, label='Average Score of last 100 games')
    plot1.set_xlabel('Episode')
    plot1.set_ylabel('Score')
    plot1.set_title("Score and average score")
    plot1.legend()


    plot2.plot(x_values, data3,label='Time of each episode',color='red')
    plot2.plot(x_values,data4, label='Average time of the last 100 games', color='black')
    plot2.set_xlabel('Episode')
    plot2.set_ylabel('Seconds')
    plot2.set_title('Time and average time')
    plot2.legend()

    plt.savefig(plotname+".png")
    plt.show()

   