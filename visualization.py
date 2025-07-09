"""
DEM Visualization and Plotting

This module provides comprehensive visualization tools for lunar DEMs including
2D plots, 3D surface rendering, and interactive visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


class DEMVisualizer:
    """
    DEM visualization class for lunar surface data.
    
    Provides methods for:
    - 2D contour and surface plots
    - 3D surface rendering
    - Interactive visualizations
    - Statistical plots
    """
    
    def __init__(self):
        """Initialize the DEM visualizer."""
        self.colormaps = {
            'terrain': 'terrain',
            'elevation': 'viridis',
            'difference': 'RdBu_r',
            'gradient': 'hot',
            'lunar': 'gray'
        }
        
    def plot_contours(self, dem, levels=20, title="Lunar DEM Contours", 
                     save_path=None, figsize=(10, 8)):
        """
        Create contour plot of the DEM.
        
        Args:
            dem (numpy.ndarray): DEM array
            levels (int): Number of contour levels
            title (str): Plot title
            save_path (str): Path to save plot
            figsize (tuple): Figure size
            
        Returns:
            matplotlib.figure.Figure: Figure object
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create contour plot
        contour = ax.contour(dem, levels=levels, colors='black', alpha=0.7)
        contourf = ax.contourf(dem, levels=levels, cmap=self.colormaps['terrain'])
        
        # Add colorbar
        cbar = plt.colorbar(contourf, ax=ax)
        cbar.set_label('Elevation')
        
        # Customize plot
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_aspect('equal')
        
        # Add contour labels
        ax.clabel(contour, inline=True, fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        plt.show()
        return fig
    
    def plot_surface(self, dem, title="Lunar DEM Surface", save_path=None, 
                    figsize=(12, 8)):
        """
        Create 2D surface plot of the DEM.
        
        Args:
            dem (numpy.ndarray): DEM array
            title (str): Plot title
            save_path (str): Path to save plot
            figsize (tuple): Figure size
            
        Returns:
            matplotlib.figure.Figure: Figure object
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create surface plot
        im = ax.imshow(dem, cmap=self.colormaps['terrain'], aspect='equal')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Elevation')
        
        # Customize plot
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        plt.show()
        return fig
    
    def plot_3d_surface(self, dem, title="Lunar DEM 3D Surface", save_path=None,
                       figsize=(12, 10), elevation_factor=1.0):
        """
        Create 3D surface plot of the DEM.
        
        Args:
            dem (numpy.ndarray): DEM array
            title (str): Plot title
            save_path (str): Path to save plot
            figsize (tuple): Figure size
            elevation_factor (float): Factor to scale elevation for visualization
            
        Returns:
            matplotlib.figure.Figure: Figure object
        """
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Create coordinate grids
        y, x = np.mgrid[0:dem.shape[0]:1, 0:dem.shape[1]:1]
        
        # Scale elevation for better visualization
        z = dem * elevation_factor
        
        # Create 3D surface
        surf = ax.plot_surface(x, y, z, cmap=self.colormaps['terrain'],
                              linewidth=0, antialiased=True, alpha=0.8)
        
        # Customize plot
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_zlabel('Elevation')
        
        # Add colorbar
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        
        # Set viewing angle
        ax.view_init(elev=30, azim=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        plt.show()
        return fig
    
    def visualize_3d(self, dem, save_path=None, title="Lunar DEM 3D Visualization",
                    colormap='viridis', show_contours=True):
        """
        Create interactive 3D visualization using Plotly.
        
        Args:
            dem (numpy.ndarray): DEM array
            save_path (str): Path to save HTML file
            title (str): Plot title
            colormap (str): Colormap name (must be a valid Plotly colorscale, e.g., 'viridis', 'earth', 'jet')
            show_contours (bool): Whether to show contour lines
            
        Returns:
            plotly.graph_objects.Figure: Interactive 3D figure
        """
        # Create coordinate grids
        y, x = np.mgrid[0:dem.shape[0]:1, 0:dem.shape[1]:1]
        
        # Create 3D surface plot
        fig = go.Figure()
        
        # Add surface
        surface = go.Surface(
            x=x, y=y, z=dem,
            colorscale=colormap,
            name='DEM Surface',
            showscale=True,
            colorbar=dict(title="Elevation")
        )
        fig.add_trace(surface)
        
        # Add contour lines if requested
        if show_contours:
            contours = go.Surface(
                x=x, y=y, z=dem,
                colorscale=[[0, 'black'], [1, 'black']],
                showscale=False,
                opacity=0.3,
                contours=dict(
                    z=dict(
                        show=True,
                        usecolormap=False,
                        highlightcolor="black",
                        project=dict(z=True)
                    )
                )
            )
            fig.add_trace(contours)
        
        # Update layout
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='X Coordinate',
                yaxis_title='Y Coordinate',
                zaxis_title='Elevation',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            width=800,
            height=600
        )
        
        if save_path:
            fig.write_html(save_path)
            
        fig.show()
        return fig
    
    def plot_comparison(self, dem1, dem2, titles=None, save_path=None):
        """
        Create side-by-side comparison of two DEMs.
        
        Args:
            dem1 (numpy.ndarray): First DEM
            dem2 (numpy.ndarray): Second DEM
            titles (list): Titles for the plots
            save_path (str): Path to save plot
            
        Returns:
            matplotlib.figure.Figure: Figure object
        """
        if titles is None:
            titles = ["DEM 1", "DEM 2"]
            
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot first DEM
        im1 = axes[0].imshow(dem1, cmap=self.colormaps['terrain'])
        axes[0].set_title(titles[0], fontsize=12, fontweight='bold')
        axes[0].set_xlabel('X Coordinate')
        axes[0].set_ylabel('Y Coordinate')
        plt.colorbar(im1, ax=axes[0])
        
        # Plot second DEM
        im2 = axes[1].imshow(dem2, cmap=self.colormaps['terrain'])
        axes[1].set_title(titles[1], fontsize=12, fontweight='bold')
        axes[1].set_xlabel('X Coordinate')
        axes[1].set_ylabel('Y Coordinate')
        plt.colorbar(im2, ax=axes[1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        plt.show()
        return fig
    
    def plot_difference(self, dem1, dem2, title="DEM Difference", save_path=None):
        """
        Plot the difference between two DEMs.
        
        Args:
            dem1 (numpy.ndarray): First DEM
            dem2 (numpy.ndarray): Second DEM
            title (str): Plot title
            save_path (str): Path to save plot
            
        Returns:
            matplotlib.figure.Figure: Figure object
        """
        difference = dem1 - dem2
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create difference plot
        im = ax.imshow(difference, cmap=self.colormaps['difference'], 
                      vmin=-np.std(difference), vmax=np.std(difference))
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Elevation Difference')
        
        # Customize plot
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        plt.show()
        return fig
    
    def plot_statistics(self, dem, save_path=None):
        """
        Create statistical plots for the DEM.
        
        Args:
            dem (numpy.ndarray): DEM array
            save_path (str): Path to save plot
            
        Returns:
            matplotlib.figure.Figure: Figure object
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Histogram
        axes[0, 0].hist(dem.flatten(), bins=50, alpha=0.7, color='skyblue')
        axes[0, 0].set_title('Elevation Distribution')
        axes[0, 0].set_xlabel('Elevation')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Gradient magnitude
        grad_x, grad_y = np.gradient(dem)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        im1 = axes[0, 1].imshow(grad_mag, cmap=self.colormaps['gradient'])
        axes[0, 1].set_title('Gradient Magnitude')
        plt.colorbar(im1, ax=axes[0, 1])
        
        # Elevation profile (middle row)
        middle_row = dem[dem.shape[0]//2, :]
        axes[1, 0].plot(middle_row, 'b-', linewidth=2)
        axes[1, 0].set_title('Elevation Profile (Middle Row)')
        axes[1, 0].set_xlabel('X Coordinate')
        axes[1, 0].set_ylabel('Elevation')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Elevation profile (middle column)
        middle_col = dem[:, dem.shape[1]//2]
        axes[1, 1].plot(middle_col, 'r-', linewidth=2)
        axes[1, 1].set_title('Elevation Profile (Middle Column)')
        axes[1, 1].set_xlabel('Y Coordinate')
        axes[1, 1].set_ylabel('Elevation')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        plt.show()
        return fig
    
    def create_animation(self, dem_sequence, save_path=None, title="DEM Animation"):
        """
        Create animation from sequence of DEMs.
        
        Args:
            dem_sequence (list): List of DEM arrays
            save_path (str): Path to save animation
            title (str): Animation title
            
        Returns:
            matplotlib.animation.Animation: Animation object
        """
        from matplotlib.animation import FuncAnimation
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Initialize plot
        im = ax.imshow(dem_sequence[0], cmap=self.colormaps['terrain'])
        ax.set_title(f"{title} - Frame 0")
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Elevation')
        
        def animate(frame):
            im.set_array(dem_sequence[frame])
            ax.set_title(f"{title} - Frame {frame}")
            return [im]
        
        anim = FuncAnimation(fig, animate, frames=len(dem_sequence), 
                           interval=500, blit=True)
        
        if save_path:
            anim.save(save_path, writer='pillow', fps=2)
            
        plt.show()
        return anim
    
    def plot_feature_analysis(self, dem, feature_locations=None, save_path=None):
        """
        Plot DEM with highlighted features.
        
        Args:
            dem (numpy.ndarray): DEM array
            feature_locations (list): List of feature coordinates [(x, y), ...]
            save_path (str): Path to save plot
            
        Returns:
            matplotlib.figure.Figure: Figure object
        """
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Plot DEM
        im = ax.imshow(dem, cmap=self.colormaps['terrain'])
        
        # Highlight features if provided
        if feature_locations:
            for i, (x, y) in enumerate(feature_locations):
                ax.plot(x, y, 'ro', markersize=8, markeredgecolor='white', 
                       markeredgewidth=2, label=f'Feature {i+1}')
                
                # Add feature number
                ax.annotate(f'{i+1}', (x+2, y+2), color='white', 
                           fontsize=10, fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Elevation')
        
        # Customize plot
        ax.set_title('Lunar DEM with Feature Analysis', fontsize=14, fontweight='bold')
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        
        if feature_locations:
            ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        plt.show()
        return fig 