import numpy as np
import plotly.graph_objects as go
import plotly.figure_factory as ff
import plotly.subplots as sp

def plot_lr_curve(history):
    lr = history["lr"]

    fig = go.Figure()
        
    fig.add_trace(go.Scatter(x=np.arange(1, len(lr)+1), mode="lines", y=lr, marker=dict(color="indianred")))
    
    fig.update_layout(title_text="LR per Epochs", yaxis_title="LR", xaxis_title="Epochs", template="plotly_white")
    fig.show()

def plot_loss_curve(history):
    training = history["train"]["loss"]
    validation= history["valid"]["loss"]

    fig = go.Figure()
        
    fig.add_trace(go.Scatter(x=np.arange(1, len(training)+1), mode="lines+markers", y=training, marker=dict(color="dodgerblue"), name="Train"))
    fig.add_trace(go.Scatter(x=np.arange(1, len(validation)+1), mode="lines+markers", y=validation, marker=dict(color="darkorange"), name="Valid"))
    
    fig.update_layout(title_text="Loss vs. Epochs", yaxis_title="Loss", xaxis_title="Epochs", template="plotly_white")
    fig.show()

def plot_acc_curve(history):
    training = history["train"]["acc"]
    validation = history["valid"]["acc"]

    fig = go.Figure()
        
    fig.add_trace(go.Scatter(x=np.arange(1, len(training)+1), mode="lines+markers", y=training, marker=dict(color="dodgerblue"), name="Train"))
    fig.add_trace(go.Scatter(x=np.arange(1, len(validation)+1), mode="lines+markers", y=validation, marker=dict(color="darkorange"), name="Valid"))
    
    fig.update_layout(title_text="Acc vs. Epochs", yaxis_title="Acc", xaxis_title="Epochs", template="plotly_white")
    fig.show()

def plot_confusion_matrix(confusion_matrix, class_names):
    z = confusion_matrix.astype(int)
    z_text = confusion_matrix.astype(str)

    layout = {
        "title": "Confusion Matrix", 
        "xaxis": {"title": "Predicted value"}, 
        "yaxis": {"title": "Real value"}
    }

    fig = ff.create_annotated_heatmap(z, x=class_names, y=class_names, annotation_text=z_text, colorscale="Viridis")
    
    fig.update_layout(layout)
    fig.show()

def plot_prediction(image, probabilities, class_names):
    fig = sp.make_subplots(rows=1, cols=2)

    fig.add_trace(go.Image(z=image.convert('RGB'), name="Image"), row=1, col=1)
    fig.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
    fig.add_trace(go.Bar(x=class_names, y=probabilities, marker=dict(color="limegreen"), name="Probabilities"), row=1, col=2)

    fig.update_layout(title_text="Predictions")
    fig.show()