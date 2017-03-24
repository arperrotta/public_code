'''This is a function to make scatter plots that Eric Alm likes. Enjoy!
	you need to have seaborn (sns), and matplot.pyplot (plt) imported to run
	data = a pandas dataframe with the following columns:
		yName - the data you want to plot, e.g. alpha diversity for each sample
		xName - how you want to visualize your data, e.g. two different time points
		colorName - how you want to color your data at each time point, e.g. disease verses control
	These column names are also input to the function
	NOTE: For visualizing these plots see Scatter_plots_Eric_likes.ipynb'''
import seaborn as sns
import matplotlib.pyplot as plt

def scatter4eric(data,xName,yName,colorName):
	cx=sns.stripplot(x=xName, y=yName, hue=colorName, data=data,split=True,jitter=True)
	sns.boxplot(x=xName, y=yName, hue=colorName,data=data,
               	saturation=0.8,whis=0, fliersize=0,width=0.8,**{'showcaps':False})
	plt.setp(cx.artists, alpha=0.5, linewidth=0, fill=False, edgecolor="k")
	cx.legend_.remove()
	handles, labels = cx.get_legend_handles_labels()
	l = plt.legend(handles[0:2], labels[0:2], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
	return(cx)
