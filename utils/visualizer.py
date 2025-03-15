import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

class Visualizer:
    def __init__(self):
        self.color_scheme = {
            'primary': '#4B8BBE',
            'secondary': '#FFD43B',
            'accent': '#306998',
            'text': '#2C3E50'
        }

    def create_correlation_matrix(self, df: pd.DataFrame) -> go.Figure:
        """Create a professional correlation matrix heatmap with enhanced visualization."""
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.shape[1] < 2:
            # Not enough numeric columns
            fig = go.Figure()
            fig.update_layout(
                title="Not enough numeric columns for correlation analysis.",
                plot_bgcolor='white',
                paper_bgcolor='white',
                font={'color': self.color_scheme['text'], 'size': 14},
                height=400
            )
            return fig

        # Calculate correlation matrix
        corr_matrix = numeric_df.corr().round(2)
        
        # Create mask for upper triangle (to avoid redundancy)
        mask = np.zeros_like(corr_matrix, dtype=bool)
        mask[np.triu_indices_from(mask, k=1)] = True
        
        # Create a custom colorscale with professional colors
        colorscale = [
            [0.0, 'rgba(220, 20, 60, 1)'],      # Strong negative: Crimson
            [0.25, 'rgba(220, 20, 60, 0.5)'],   # Moderate negative: Lighter Crimson
            [0.5, 'rgba(240, 240, 240, 1)'],    # Neutral: Light gray
            [0.75, 'rgba(65, 105, 225, 0.5)'],  # Moderate positive: Lighter Royal Blue
            [1.0, 'rgba(65, 105, 225, 1)']      # Strong positive: Royal Blue
        ]
        
        # Create heatmap with enhanced styling
        heatmap = go.Heatmap(
            z=corr_matrix,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            zmin=-1, 
            zmax=1,
            colorscale=colorscale,
            colorbar=dict(
                title="Correlation",
                titleside="right",
                titlefont=dict(size=14),
                tickvals=[-1, -0.5, 0, 0.5, 1],
                ticktext=["-1.0", "-0.5", "0.0", "0.5", "1.0"],
            ),
            hovertemplate='%{y} & %{x}: %{z:.2f}<extra></extra>'
        )
        
        # Create annotations for correlation values
        annotations = []
        for i, row in enumerate(corr_matrix.index):
            for j, col in enumerate(corr_matrix.columns):
                value = corr_matrix.iloc[i, j]
                
                # Determine text color based on background intensity for better readability
                text_color = 'white' if abs(value) > 0.4 else 'black'
                
                # Make strong correlations bold
                font_weight = 'bold' if abs(value) > 0.7 else 'normal'
                
                annotations.append(dict(
                    x=col,
                    y=row,
                    text=str(value),
                    font=dict(
                        color=text_color,
                        size=11,
                        family="Arial",
                        weight=font_weight
                    ),
                    showarrow=False
                ))
        
        # Create the figure with the heatmap
        fig = go.Figure(data=heatmap)
        
        # Add annotations
        fig.update_layout(annotations=annotations)
        
        # Enhance the layout
        fig.update_layout(
            title={
                'text': 'Feature Correlation Matrix',
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': {'size': 18, 'color': self.color_scheme['text']}
            },
            plot_bgcolor='white',
            paper_bgcolor='white',
            height=max(400, 250 + 30 * len(corr_matrix.columns)),  # Dynamic height based on number of columns
            width=max(500, 250 + 30 * len(corr_matrix.columns)),   # Dynamic width based on number of columns
            margin=dict(l=60, r=60, t=80, b=80),
            xaxis=dict(
                side='bottom',
                tickangle=45,
                tickfont={'size': 12},
                gridcolor='lightgray',
                linecolor='lightgray',
                zeroline=False
            ),
            yaxis=dict(
                side='left',
                tickfont={'size': 12},
                gridcolor='lightgray',
                linecolor='lightgray',
                zeroline=False,
                autorange='reversed'
            )
        )
        
        return fig
