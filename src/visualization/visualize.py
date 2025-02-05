#%%
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator,FormatStrFormatter,MaxNLocator


mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['text.usetex'] = True


# %%
# x values
num_examples = [1,5,10,20]

# baseline results for 8b model
bleu_8b_baseline = 0.06
chrf_8b_baseline = 1.28

# baseline results for 70b model
bleu_70b_baseline = 0.35
chrf_70b_baseline = 7.92

# y values for 8b model
bleu_8b_results = [0.07,0.13,0.18,0.14]
chrf_8b_results = [1.84,2.89,4.07,3.59]

# y values for 70b model
bleu_70b_results = [0.2,0.26,0.23,0.37]
chrf_70b_results = [4.6,5.45,6.32,8.68]

# create figure
fig,axes = plt.subplots(nrows=2,ncols=2,figsize=(10,6), sharex=True)

# set x axis, grid, and only use outer labels
for ax in axes.flat:
    ax.set_xticks([1, 5, 10, 20])
    ax.grid(True,which='major',axis='both',alpha=0.3)
    ax.label_outer()

# set titles on top subfigures
axes[0,0].set_title(r"Llama-3.1-8B-Instruct")
axes[0,1].set_title(r'Llama-3.1-70B-Instruct')

# set y-labels on left subfigures
axes[0,0].set_ylabel(r'BLEU($\uparrow$)')
axes[1,0].set_ylabel(r'ChrF($\uparrow$)')

# set x-labels on bottom subfigures
axes[1,0].set_xlabel(r'\# examples')
axes[1,1].set_xlabel(r'\# examples')

# plot top-left subfigure
axes[0,0].plot(num_examples, bleu_8b_results)
axes[0,0].axhline(y=bleu_8b_baseline, color='red', linestyle='dashed')

# plot top-right subfigure
axes[0,1].plot(num_examples, bleu_70b_results)
axes[0,1].axhline(y=bleu_70b_baseline, color='red', linestyle='dashed')

# plot bottom-left subfigure
axes[1,0].plot(num_examples, chrf_8b_results)
axes[1,0].axhline(y=chrf_8b_baseline, color='red', linestyle='dashed')

# plot bottom-right subfigure
axes[1,1].plot(num_examples, chrf_70b_results)
axes[1,1].axhline(y=chrf_70b_baseline, color='red', linestyle='dashed', label='Zero-shot baseline')

# set legend in bottom-right subfigure
axes[1,1].legend(loc='lower right')

# share y-axis for top subfigures
axes[0,0].sharey(axes[0,1])
axes[0,0].autoscale()

# share y-axis for bottom subfigures
axes[1,0].sharey(axes[1,1])
axes[1,0].autoscale()

# remove unnecessary space between figures
fig.tight_layout()

# %%

fig.savefig('few-shot-results.png')

# %%
