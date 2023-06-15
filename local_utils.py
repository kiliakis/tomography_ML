import matplotlib.pyplot as plt


def get_best_model_timestamp(path, model='enc'):
    from sort_trial_summaries import extract_trials
    header, rows = extract_trials(path)
    for row in rows:
        if model in row[header.index('model')]:
            return row[header.index('date')]


def plot_loss(lines, title='', figname=None):
    plt.figure()
    plt.title(title)
    for line in lines.keys():
        if 'val' in line.lower():
            marker = 'x'
        else:
            marker = '.'
        plt.semilogy(lines[line], marker=marker, label=line)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(ncol=2)
    plt.tight_layout()
    if figname:
        plt.savefig(figname, dpi=300)
        plt.close()
