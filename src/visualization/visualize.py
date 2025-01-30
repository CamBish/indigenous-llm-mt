#%%
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator,FormatStrFormatter,MaxNLocator


mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'Arial'

fig = plt.figure(figsize=(10,6))
ax = plt.axes((0.1,0.1,0.5,0.8))

ax.set_xticks([0.2,1])
ax.set_xticklabels(['1','5'])

a=ax.yaxis.get_major_locator()
b=ax.yaxis.get_major_formatter()

ax.grid(True,which='major',axis='both',alpha=0.3)

# %%
num_examples = [1,5,10,20]

bleu_8b_baseline = 0.06
chrf_8b_baseline = 1.28

bleu_70b_baseline = 0.35
chrf_70b_baseline = 7.92

bleu_8b_results = [0.07,0.13,0.18,0.14]
chrf_8b_results = [1.84,2.89,4.07,3.59]

bleu_70b_results = [0.2,0.26,0.23,0.37]
chrf_70b_results = [4.6,5.45,6.32,8.68]

fig,axes = plt.subplots(nrows=2,ncols=2,figsize=(10,6), sharex=True)

for ax in axes.flat:
    ax.set_xticks([1, 5, 10, 20])
    ax.grid(True,which='major',axis='both',alpha=0.3)

axes[0,0].set_ylabel('BLEU')
axes[1,0].set_ylabel('chrF++')

axes[1,0].set_xlabel('# examples')
axes[1,1].set_xlabel('# examples')

axes[0,0].plot(num_examples, bleu_8b_results)
axes[0,1].plot(num_examples, bleu_70b_results)

axes[1,0].plot(num_examples, chrf_8b_results)
axes[1,1].plot(num_examples, chrf_70b_results)


# %%
