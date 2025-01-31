#%%
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator,FormatStrFormatter,MaxNLocator


mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['text.usetex'] = True


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

axes[0,0].set_title(r"Llama-3.1-8B-Instruct Iu $\rightarrow$ En")
axes[0,1].set_title(r'Llama-3.1-70B-Instruct Iu $\rightarrow$ En')

axes[0,0].set_ylabel('BLEU')
axes[1,0].set_ylabel('chrF++')

axes[1,0].set_xlabel(r'\# examples')
axes[1,1].set_xlabel(r'\# examples')

axes[0,0].plot(num_examples, bleu_8b_results)
axes[0,0].axhline(y=bleu_8b_baseline, color='red', linestyle='dashed')

axes[0,1].plot(num_examples, bleu_70b_results)
axes[0,1].axhline(y=bleu_70b_baseline, color='red', linestyle='dashed')

axes[1,0].plot(num_examples, chrf_8b_results)
axes[1,0].axhline(y=chrf_8b_baseline, color='red', linestyle='dashed')

axes[1,1].plot(num_examples, chrf_70b_results)
axes[1,1].axhline(y=chrf_70b_baseline, color='red', linestyle='dashed', label='Zero-shot baseline')

axes[1,1].legend(loc='lower right')

fig.tight_layout()

# %%
