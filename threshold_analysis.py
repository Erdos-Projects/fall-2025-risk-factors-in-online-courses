from sklearn.metrics import confusion_matrix, roc_auc_score, precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt

def metrics_at_thresholds(thresholds, tuned_models, X, y):
    # performance metrics for specified models at given classification thresholds
    
    # inputs: thresholds = list of classification thresholds;
    # tuned_models = dictionary of models to test;
    # X = dataframe of predictors, y = dataframe of true values.
   
    # returns: list of tuples, each tuple containing model name and metrics.

    results = []

    for thresh in thresholds:
        for name, model in tuned_models:
            y_proba = model.predict_proba(X)[:, 1]
            y_pred = (y_proba >= thresh).astype(int)
            
            # Calculate confusion matrix components
            tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
            
            # Calculate False Negative Rate (FNR) and False Positive Rate (FPR)
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0  # False Negative Rate = FN / (FN + TP)
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate = FP / (FP + TN)
            
            results.append({
                'Model': name,
                'Threshold': thresh,
                'Recall': recall_score(y, y_pred),
                'Precision': precision_score(y, y_pred),
                'F1': f1_score(y, y_pred),
                'FNR': fnr,
                'FPR': fpr
            })
    
    return results

def visualize_thresholds(df_results):
    # Visualization of dataframe of results produced by metrics_at_thresholds
    # as graphs of performance metric vs. threshold
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 8))
    axes = axes.flatten()

    for idx, metric in enumerate(['Recall', 'Precision', 'F1', 'FNR', 'FPR']):
        ax = axes[idx]
        for model in ['LR', 'RF', 'ET']:
            data = df_results[df_results['Model'] == model]
            ax.plot(data['Threshold'], data[metric], 'o-', label=model, linewidth=2)
        ax.set_xlabel('Threshold')
        ax.set_ylabel(metric)
        ax.set_title(metric)
        ax.legend()
        ax.grid(alpha=0.3)

    # Hide the extra subplot (6th position)
    axes[5].axis('off')

    plt.tight_layout()
    plt.show()