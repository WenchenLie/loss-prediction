import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
from pathlib import Path


LABEL_NAMES = ['loss1_mean']
FEATURE_NAMES = {
    '4': ['R', 'lamda', 'a1'],
    '8': ['R', 'lamda', 'a1', 'a2'],
    '12': ['R', 'lamda', 'a1', 'a2', 'a3']
}
FONTSIZE = 24

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.weight'] = 'normal'
plt.rcParams['axes.labelweight'] = 'normal'
plt.rcParams['axes.titleweight'] = 'normal'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['xtick.labelsize'] = FONTSIZE 
plt.rcParams['ytick.labelsize'] = FONTSIZE

def plot_shap(n_story: int, feature_names: list[str], model_name: str, label_to_plot: str, output_dir: Path = Path('ml_output/Regression'), annotation_text: str = ""):
    df_x = pd.read_csv(output_dir / f'{model_name}_{n_story}S_x_test.csv', index_col=0)
    df_y = pd.read_csv(output_dir / f'{model_name}_{n_story}S_y_test.csv', index_col=0)
    x_test = df_x.to_numpy()
    y_test = df_y[f'Actual_{label_to_plot}'].to_numpy()
    shap_values: np.ndarray = np.loadtxt(output_dir / f'shap_values_{model_name}_{n_story}S.txt')
    
    if len(LABEL_NAMES) > 1:
        idx = LABEL_NAMES.index(label_to_plot)
        shap_values = shap_values[:, :, idx]
        
    feature_importance = np.mean(np.abs(shap_values), axis=0)
    sort_inds = np.argsort(feature_importance)
    sorted_importance = feature_importance[sort_inds]
    
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
    feature_importance_df.sort_values(by='Importance', ascending=False, inplace=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_importance_df.to_csv(output_dir / f'{model_name}_{n_story}S_{label_to_plot}_importance.csv', index=False)

    display_feature_names = []
    for name in feature_names:
        if name == 'lamda':
            display_feature_names.append(r'$\lambda$')
        elif len(name) > 1 and name[1:].isdigit():
            display_feature_names.append(fr'${name[0]}_{{\mathrm{{{name[1:]}}}}}$')
        else:
            display_feature_names.append(fr'${name}$')

    fig = plt.figure(figsize=(10, 5))
    
    scatter_cmap = mcolors.LinearSegmentedColormap.from_list('shap_scatter', ['#1E88E5', '#FF0052'])
    bar_cmap = mcolors.LinearSegmentedColormap.from_list('shap_bar', ['#cce0ff', '#ffb3cc'])

    shap.summary_plot(
        shap_values, x_test, feature_names=display_feature_names, 
        show=False, cmap=scatter_cmap, alpha=1.0, plot_size=None,
        color_bar=False 
    )
    
    ax1 = plt.gca()
    ax2 = ax1.twiny()
    
    y_pos = np.arange(len(sorted_importance))
    bar_colors = bar_cmap(np.linspace(0, 1, len(sorted_importance)))
    
    ax2.barh(y_pos, sorted_importance, height=0.6, color=bar_colors, alpha=0.8, zorder=1)
    
    ax2.set_zorder(1)
    ax1.set_zorder(10)
    ax1.patch.set_visible(False)
    
    ax2.set_xlabel('Mean |SHAP value|', fontsize=FONTSIZE, labelpad=10)
    ax2.set_xlim(0, max(sorted_importance) * 1.2)
    ax2.tick_params(axis='x', direction='in', width=1.5, length=5, labelsize=FONTSIZE, top=True, labeltop=True, bottom=False, labelbottom=False)
    
    ax1.set_xlabel(r'SHAP value (impact on $\mathit{LR}$)', fontsize=FONTSIZE, labelpad=2)
    ax1.tick_params(axis='x', direction='in', width=1.5, length=5, labelsize=FONTSIZE, bottom=True, labelbottom=True, top=False, labeltop=False)
    
    ax1.tick_params(axis='y', length=0, labelsize=FONTSIZE, left=False)
    
    num_features = len(feature_names)
    ax1.set_ylim(-0.5, num_features - 0.5)

    ax1.xaxis.grid(True, dashes=(8, 8), color='gray', alpha=0.5, zorder=0)
    ax1.axvline(x=0, color='#999999', linewidth=2, zorder=2)
    
    for ax in [ax1, ax2]:
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
            spine.set_color('black')
            
    plt.tight_layout()
    plt.subplots_adjust(right=0.85)
    
    ax1_pos = ax1.get_position()
    cax = fig.add_axes([ax1_pos.x1 + 0.02, ax1_pos.y0, 0.025, ax1_pos.height])
    
    sm = plt.cm.ScalarMappable(cmap=scatter_cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    
    cb = fig.colorbar(sm, cax=cax)
    cb.set_ticks([0, 1])
    cb.set_ticklabels(['Low', 'High'])
    cb.ax.tick_params(labelsize=FONTSIZE, length=0)
    
    cb.outline.set_linewidth(1.5)
    cb.outline.set_color('black')
    cb.set_label('Feature value', fontsize=FONTSIZE, labelpad=5)
            
    if annotation_text:
        ax1.text(0.97, 0.03, annotation_text, transform=ax1.transAxes,
                 fontsize=FONTSIZE, ha='right', va='bottom', zorder=20)
            
    plt.savefig(output_dir / f'shap_combined_{model_name}_{n_story}S_{label_to_plot}.tiff', dpi=600, format='tiff', bbox_inches='tight', pil_kwargs={"compression": "tiff_lzw"})
    plt.show()

if __name__ == "__main__":
    i = 0
    for case_ in ['Classification', 'Regression']:
        for n_story in [4, 8, 12]:
            feature_names = FEATURE_NAMES[str(n_story)]
            output_dir = Path(f'ml_output/{case_}')
            No = ['a', 'b', 'c', 'd', 'e', 'f'][i]
            step = 'Step-I' if case_ == 'Classification' else 'Step-II'
            annotation = f'({No}) {step} ({n_story}-story)'
            plot_shap(n_story, feature_names, 'XGBoost', label_to_plot='loss1_mean', annotation_text=annotation, output_dir=output_dir)
            i += 1