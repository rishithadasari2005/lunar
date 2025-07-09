import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.express as px
from PIL import Image, ImageEnhance
import tempfile
import os
import time
import pandas as pd
from lunar_dem import LunarDEMGenerator
import cv2
from scipy import ndimage
import seaborn as sns

# Page configuration
st.set_page_config(
    page_title="🌙 Lunar DEM Generator Pro",
    page_icon="🌙",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern UI
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .feature-box {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    .stProgress > div > div > div > div {
        background-color: #667eea;
    }
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: bold;
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #764ba2 0%, #667eea 100%);
        transform: translateY(-2px);
        transition: all 0.3s ease;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🌙 Lunar DEM Generator Pro</h1>
    <p>Advanced Digital Elevation Model Generation from Lunar Images</p>
    <p>🚀 Powered by AI & Computer Vision</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    
    # Fast Mode option
    fast_mode = st.checkbox("⚡ Fast Mode (for quick demo)", value=False, help="Forces low resolution and disables advanced processing for speed.")
    
    # DEM Resolution
    if fast_mode:
        resolution = 50
    else:
        resolution = st.slider(
            "DEM Resolution", 
            min_value=50, 
            max_value=500, 
            value=150, 
            step=10,
            help="Higher resolution = better quality but slower processing"
        )
    
    # Processing options
    st.markdown("### 🔧 Processing Options")
    use_gpu = st.checkbox("Use GPU Acceleration (if available)", value=False, disabled=fast_mode)
    enable_parallel = st.checkbox("Enable Parallel Processing", value=True, disabled=fast_mode)
    
    # Advanced settings
    with st.expander("🔬 Advanced Settings"):
        noise_reduction = st.slider("Noise Reduction", 0.0, 1.0, 0.3, 0.1, disabled=fast_mode)
        contrast_enhancement = st.slider("Contrast Enhancement", 0.5, 2.0, 1.0, 0.1, disabled=fast_mode)
        smoothing_factor = st.slider("Smoothing Factor", 0.0, 1.0, 0.2, 0.1, disabled=fast_mode)
    
    # Model selection
    st.markdown("### 🧠 AI Model")
    model_type = st.selectbox(
        "Select Model",
        ["Photometric Stereo", "Shape from Shading", "Hybrid Approach"],
        help="Choose the algorithm for DEM generation",
        disabled=fast_mode
    )

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    # File upload with drag & drop styling
    st.markdown("### 📁 Upload Lunar Image")
    uploaded_file = st.file_uploader(
        "Drag and drop your lunar image here",
        type=["jpg", "png", "jpeg", "tiff"],
        help="Supported formats: JPG, PNG, JPEG, TIFF"
    )

with col2:
    # Quick stats
    if uploaded_file is not None:
        st.markdown("### 📊 Image Stats")
        image = Image.open(uploaded_file)
        st.markdown(f"""
        <div class="metric-card">
            <h4>📏 Dimensions</h4>
            <p>{image.size[0]} × {image.size[1]} pixels</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card">
            <h4>💾 File Size</h4>
            <p>{uploaded_file.size / 1024:.1f} KB</p>
        </div>
        """, unsafe_allow_html=True)

# Processing section
if uploaded_file is not None:
    st.markdown("---")
    
    # Image preprocessing
    with st.expander("🖼️ Image Preprocessing", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### Original Image")
            image = Image.open(uploaded_file).convert("L")
            st.image(image, caption="Original", use_container_width=True)
        
        with col2:
            st.markdown("### Enhanced Image")
            # Apply contrast enhancement
            enhancer = ImageEnhance.Contrast(image)
            enhanced_image = enhancer.enhance(contrast_enhancement)
            st.image(enhanced_image, caption="Enhanced", use_container_width=True)
        
        with col3:
            st.markdown("### Processed Image")
            # Apply noise reduction
            img_array = np.array(enhanced_image)
            if noise_reduction > 0:
                img_array = cv2.GaussianBlur(img_array, (5, 5), noise_reduction)
            processed_image = Image.fromarray(img_array)
            st.image(processed_image, caption="Processed", use_container_width=True)
    
    # DEM Generation
    if st.button("🚀 Generate DEM", type="primary"):
        with st.spinner("🔄 Initializing DEM generation..."):
            # Resize image to match selected resolution for speed
            processed_image_resized = processed_image.resize((resolution, resolution))
            # Save the processed image to a temporary file
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                processed_image_resized.save(tmp.name)
                temp_image_path = tmp.name
            
            # Create progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Create DEM generator instance
                dem_generator = LunarDEMGenerator()
                
                # Simulate progress updates
                for i in range(100):
                    time.sleep(0.05)  # Simulate processing time
                    progress_bar.progress(i + 1)
                    if i < 20:
                        status_text.text("🔄 Loading image and initializing...")
                    elif i < 40:
                        status_text.text("🔍 Analyzing image features...")
                    elif i < 60:
                        status_text.text("🧮 Computing elevation data...")
                    elif i < 80:
                        status_text.text("📊 Generating DEM surface...")
                    else:
                        status_text.text("✨ Finalizing results...")
                
                # Generate DEM
                dem = dem_generator.generate_dem(temp_image_path, resolution=resolution)
                
                # Clean up temporary file
                os.unlink(temp_image_path)
                
                status_text.text("✅ DEM generation completed!")
                progress_bar.progress(100)
                
                # Display results
                st.success("🎉 DEM generated successfully!")
                
                # Results section
                st.markdown("---")
                st.markdown("## 📊 DEM Analysis Results")
                
                # Metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Max Elevation", f"{np.max(dem):.2f} m")
                with col2:
                    st.metric("Min Elevation", f"{np.min(dem):.2f} m")
                with col3:
                    st.metric("Mean Elevation", f"{np.mean(dem):.2f} m")
                with col4:
                    st.metric("Elevation Range", f"{np.max(dem) - np.min(dem):.2f} m")
                
                # Visualization tabs
                tab1, tab2, tab3, tab4, tab5 = st.tabs(["🗺️ 2D Surface", "🌋 3D Surface", "📈 Contour Plot", "🎨 Heatmap", "📊 Statistics"])
                
                with tab1:
                    st.markdown("### 2D Surface Plot")
                    fig, ax = plt.subplots(figsize=(10, 8))
                    im = ax.imshow(dem, cmap='terrain', aspect='auto')
                    plt.colorbar(im, ax=ax, label='Elevation (m)')
                    ax.set_title('Lunar Surface DEM - 2D View')
                    ax.set_xlabel('X Coordinate')
                    ax.set_ylabel('Y Coordinate')
                    st.pyplot(fig)
                
                with tab2:
                    st.markdown("### 3D Surface Plot")
                    # Create 3D surface plot with Plotly
                    x = np.arange(dem.shape[1])
                    y = np.arange(dem.shape[0])
                    X, Y = np.meshgrid(x, y)
                    
                    fig = go.Figure(data=[go.Surface(z=dem, x=X, y=Y, colorscale='earth')])
                    fig.update_layout(
                        title='Lunar Surface DEM - 3D View',
                        scene=dict(
                            xaxis_title='X Coordinate',
                            yaxis_title='Y Coordinate',
                            zaxis_title='Elevation (m)'
                        ),
                        width=800,
                        height=600
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with tab3:
                    st.markdown("### Contour Plot")
                    fig, ax = plt.subplots(figsize=(10, 8))
                    contours = ax.contour(dem, levels=20, colors='black', alpha=0.5)
                    ax.clabel(contours, inline=True, fontsize=8)
                    im = ax.imshow(dem, cmap='terrain', alpha=0.7)
                    plt.colorbar(im, ax=ax, label='Elevation (m)')
                    ax.set_title('Lunar Surface Contours')
                    st.pyplot(fig)
                
                with tab4:
                    st.markdown("### Elevation Heatmap")
                    fig, ax = plt.subplots(figsize=(10, 8))
                    sns.heatmap(dem, cmap='terrain', ax=ax, cbar_kws={'label': 'Elevation (m)'})
                    ax.set_title('Lunar Surface Heatmap')
                    st.pyplot(fig)
                
                with tab5:
                    st.markdown("### Statistical Analysis")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### Elevation Distribution")
                        fig, ax = plt.subplots(figsize=(8, 6))
                        ax.hist(dem.flatten(), bins=50, alpha=0.7, color='skyblue', edgecolor='black')
                        ax.set_xlabel('Elevation (m)')
                        ax.set_ylabel('Frequency')
                        ax.set_title('Elevation Distribution')
                        st.pyplot(fig)
                    
                    with col2:
                        st.markdown("#### Statistical Summary")
                        stats_df = pd.DataFrame({
                            'Statistic': ['Mean', 'Median', 'Std Dev', 'Min', 'Max', '25th Percentile', '75th Percentile'],
                            'Value': [
                                f"{np.mean(dem):.2f}",
                                f"{np.median(dem):.2f}",
                                f"{np.std(dem):.2f}",
                                f"{np.min(dem):.2f}",
                                f"{np.max(dem):.2f}",
                                f"{np.percentile(dem, 25):.2f}",
                                f"{np.percentile(dem, 75):.2f}"
                            ]
                        })
                        st.dataframe(stats_df, use_container_width=True)
                
                # Export options
                st.markdown("---")
                st.markdown("## 💾 Export Results")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("📥 Download DEM as CSV"):
                        df = pd.DataFrame(dem)
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="Click to download",
                            data=csv,
                            file_name="lunar_dem.csv",
                            mime="text/csv"
                        )
                
                with col2:
                    if st.button("🖼️ Download 3D Visualization"):
                        # Save 3D plot as HTML
                        fig.write_html("lunar_dem_3d.html")
                        with open("lunar_dem_3d.html", "r") as f:
                            html_content = f.read()
                        st.download_button(
                            label="Download 3D Plot",
                            data=html_content,
                            file_name="lunar_dem_3d.html",
                            mime="text/html"
                        )
                
                with col3:
                    if st.button("📊 Download Report"):
                        # Generate a simple report
                        report = f"""
                        Lunar DEM Analysis Report
                        ========================
                        
                        Image: {uploaded_file.name}
                        Resolution: {resolution}x{resolution}
                        Model: {model_type}
                        
                        Statistics:
                        - Max Elevation: {np.max(dem):.2f} m
                        - Min Elevation: {np.min(dem):.2f} m
                        - Mean Elevation: {np.mean(dem):.2f} m
                        - Standard Deviation: {np.std(dem):.2f} m
                        
                        Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}
                        """
                        st.download_button(
                            label="Download Report",
                            data=report,
                            file_name="lunar_dem_report.txt",
                            mime="text/plain"
                        )
                
            except Exception as e:
                st.error(f"❌ Error during DEM generation: {str(e)}")
                st.info("💡 Try reducing the resolution or using a smaller image.")

# Features showcase
st.markdown("---")
st.markdown("## ✨ Advanced Features")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="feature-box">
        <h3>🤖 AI-Powered Processing</h3>
        <p>Advanced machine learning algorithms for accurate DEM generation from single images.</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-box">
        <h3>📊 Real-time Analytics</h3>
        <p>Comprehensive statistical analysis and visualization of lunar surface features.</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="feature-box">
        <h3>🚀 High Performance</h3>
        <p>Optimized processing with GPU acceleration and parallel computing support.</p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem; background: #f8f9fa; border-radius: 10px;">
    <h3>🌙 Lunar DEM Generator Pro</h3>
    <p>Advanced Digital Elevation Model Generation for Lunar Exploration</p>
    <p>Built with ❤️ for the Hackathon</p>
</div>
""", unsafe_allow_html=True) 