import numpy as np
import os
from utils import load_data, load_model
import plotly.graph_objects as go
import plotly.io as pio

def plot_3d_gp_surface(func_id, X_train, y_train, model, save_dir="images"):
    """Generate and save a 3D surface plot of the GP mean prediction."""
    x1 = np.linspace(0, 1, 100)
    x2 = np.linspace(0, 1, 100)
    X1, X2 = np.meshgrid(x1, x2)
    X_grid = np.vstack([X1.ravel(), X2.ravel()]).T

    Z_mean, _ = model.predict(X_grid)
    Z_mean = Z_mean.reshape(X1.shape)

    # Create 3D surface plot
    surface = go.Surface(z=Z_mean, x=X1, y=X2, colorscale='Viridis')
    scatter = go.Scatter3d(
        x=X_train[:, 0], y=X_train[:, 1], z=y_train,
        mode='markers',
        marker=dict(color='red', size=4),
        name='Observations'
    )

    fig = go.Figure(data=[surface, scatter])
    fig.update_layout(
        title=f"Function {func_id} - GP Mean Prediction (3D)",
        scene=dict(
            xaxis_title='x1',
            yaxis_title='x2',
            zaxis_title='f(x1, x2)'
        )
    )

    # Save as HTML and image
    os.makedirs(save_dir, exist_ok=True)
    html_path = os.path.join(save_dir, f"function_{func_id}_surface.html")
    jpg_path = os.path.join(save_dir, f"function_{func_id}_surface.jpg")
    fig.write_html(html_path)
    pio.write_image(fig, jpg_path, format='jpg', width=800, height=600, scale=2)
    print(f"3D surface plot saved: {html_path} and {jpg_path}")

if __name__ == "__main__":
    for func_id in [1, 2]:
        X, y = load_data(func_id)
        model = load_model(func_id)

        if X is not None and y is not None and model is not None:
            plot_3d_gp_surface(func_id, X, y, model)
        else:
            print(f"Missing model or data for Function {func_id}. Skipping.")
