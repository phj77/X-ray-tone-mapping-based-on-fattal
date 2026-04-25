"""
Streamlit App for HDR Tone Mapping with Interactive Parameter Tuning
Integrates: preprocess -> tonemap -> postprocess
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

from generals.utils import load_hdr, save_ldr
from gradient_tone_mapping.tonemap import tonemap
from generals.preprocess import preprocess
from generals.postprocess import postprocess
from gradient_tone_mapping.parameters import Parameters

params = Parameters()

# ════════════════════════════════════════════════════════════════
# Streamlit Configuration
# ════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="HDR Tone Mapping",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🎨 HDR Tone Mapping Framework")
st.markdown("*Based on Liu et al., 2021 - Gradient Domain Tone Mapping for X-ray Images*")

# ════════════════════════════════════════════════════════════════
# Sidebar: Input Selection & Parameters
# ════════════════════════════════════════════════════════════════
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Input selection
    st.subheader("1️⃣ Input Selection")
    input_mode = st.radio("Choose input source:", ["Dataset", "Upload File"])
    
    hdr_image = None
    
    if input_mode == "Upload File":
        dataset_path = Path("poor_battery_images_dataset")
        
        if dataset_path.exists():
            # List all .hdr files in dataset
            hdr_files = list(dataset_path.rglob("*.hdr"))
            
            if hdr_files:
                selected_file = st.selectbox(
                    "Select HDR file:",
                    hdr_files,
                    format_func=lambda x: f"{x.parent.name}/{x.name}"
                )
                
                if selected_file:
                    hdr_image = load_hdr(str(selected_file))
                    st.success(f"✓ Loaded: {selected_file.name}")
            else:
                st.warning("⚠️ No .hdr files found in dataset")
        else:
            st.error("❌ Dataset folder not found")
    
    else:  # Upload File
        uploaded_file = st.file_uploader("Upload HDR file (.hdr)", type=["hdr"])
        
        if uploaded_file is not None:
            # Save temporarily
            temp_path = "temp_input.hdr"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            hdr_image = load_hdr(temp_path)
            st.success(f"✓ Loaded: {uploaded_file.name}")
    
    st.divider()
    
    # Processing parameters
    st.subheader("2️⃣ Processing Parameters")
    
    # Enable/Disable toggles
    col_pre, col_post = st.columns(2)
    with col_pre:
        apply_preprocess = st.checkbox("Apply Preprocess", value=False)
    with col_post:
        apply_postprocess = st.checkbox("Apply Postprocess", value=False)
    
    st.divider()
    
    gamma_tonemap = st.slider(
        "Tone Mapping Gamma (γ)",
        min_value=0.1,
        max_value=1.0,
        value=0.51,
        step=0.01,
        help="Paper recommendation: 0.43~0.51"
    )
    
    if apply_preprocess:
        gamma_pre = st.slider(
            "Pre-processing Gamma",
            min_value=0.0,
            max_value=3.0,
            value=2.2,
            step=0.1,
            help="Standard gamma correction value"
        )
    else:
        gamma_pre = None

    if apply_postprocess:
        gamma_post = st.slider(
            "Post-processing Gamma",
            min_value=0.0,
            max_value=3.0,
            value=2.2,
            step=0.1,
            help="Standard gamma correction value"
        )
    else:
        gamma_post = None
    
    st.divider()
    
    # Advanced options
    st.subheader("3️⃣ Advanced Options")
    
    show_intermediate = st.checkbox("Show intermediate steps", value=False)
    auto_save = st.checkbox("Auto-save output", value=False)

# ════════════════════════════════════════════════════════════════
# Main Processing Pipeline
# ════════════════════════════════════════════════════════════════
if hdr_image is not None:
    st.header("📊 Processing Pipeline")
    
    with st.spinner("Processing image..."):
        # Step 0: Load original
        print(f"[Pipeline] Input shape: {hdr_image.shape}, dtype: {hdr_image.dtype}")
        print(f"[Pipeline] Range: [{hdr_image.min():.4f}, {hdr_image.max():.4f}]")
        
        # Step 1: Preprocess (gamma correction) - optional
        if apply_preprocess:
            hdr_preprocessed = preprocess(hdr_image, gamma=gamma_pre)
            print(f"[Pipeline] After preprocess: shape={hdr_preprocessed.shape}, dtype={hdr_preprocessed.dtype}")
        else:
            hdr_preprocessed = hdr_image.astype(np.float32)
            print(f"[Pipeline] Preprocess skipped")
        
        # Step 2: Tone mapping
        ldr_tonemap = tonemap(hdr_preprocessed.astype(np.float32), gamma=gamma_tonemap)
        print(f"[Pipeline] After tonemap: range=[{ldr_tonemap.min():.4f}, {ldr_tonemap.max():.4f}]")
        
        # Step 3: Postprocess (inverse gamma correction) - optional
        if apply_postprocess:
            ldr_output = postprocess(ldr_tonemap, gamma=gamma_post)
            print(f"[Pipeline] After postprocess: shape={ldr_output.shape}, dtype={ldr_output.dtype}")
        else:
            ldr_output = (np.clip(ldr_tonemap, 0, 1) * 255).astype(np.uint8)
            print(f"[Pipeline] Postprocess skipped - direct scaling to uint8")
    
    # ════════════════════════════════════════════════════════════════
    # Visualization
    # ════════════════════════════════════════════════════════════════
    
    if show_intermediate:
        st.subheader("🔍 Intermediate Steps")
        
        # Build columns dynamically
        num_steps = 2  # Original + Tonemap always shown
        if apply_preprocess:
            num_steps += 1
        if apply_postprocess:
            num_steps += 1
        
        cols = st.columns(num_steps)
        col_idx = 0
        
        # Original
        with cols[col_idx]:
            st.markdown("**Original HDR**")
            fig1, ax1 = plt.subplots(figsize=(4, 4))
            hdr_display = np.clip(hdr_image / hdr_image.max(), 0, 1)
            ax1.imshow(hdr_display, cmap='gray')
            ax1.axis('off')
            st.pyplot(fig1)
            st.caption(f"Range: [{hdr_image.min():.2f}, {hdr_image.max():.2f}]")
        col_idx += 1
        
        # Preprocess
        if apply_preprocess:
            with cols[col_idx]:
                st.markdown("**After Preprocess**")
                fig2, ax2 = plt.subplots(figsize=(4, 4))
                preproc_display = np.clip(hdr_preprocessed / 255.0, 0, 1)
                ax2.imshow(preproc_display, cmap='gray')
                ax2.axis('off')
                st.pyplot(fig2)
                st.caption(f"Range: [{hdr_preprocessed.min():.2f}, {hdr_preprocessed.max():.2f}]")
            col_idx += 1
        
        # Tonemap
        with cols[col_idx]:
            st.markdown("**After Tonemap**")
            fig3, ax3 = plt.subplots(figsize=(4, 4))
            ax3.imshow(ldr_tonemap, cmap='gray')
            ax3.axis('off')
            st.pyplot(fig3)
            st.caption(f"Range: [{ldr_tonemap.min():.4f}, {ldr_tonemap.max():.4f}]")
        col_idx += 1
        
        # Postprocess
        if apply_postprocess:
            with cols[col_idx]:
                st.markdown("**After Postprocess**")
                fig4, ax4 = plt.subplots(figsize=(4, 4))
                output_display = np.clip(ldr_output / 255.0, 0, 1)
                ax4.imshow(output_display, cmap='gray')
                ax4.axis('off')
                st.pyplot(fig4)
                st.caption(f"Range: [{ldr_output.min():.2f}, {ldr_output.max():.2f}]")
    
    # ════════════════════════════════════════════════════════════════
    # Final Comparison
    # ════════════════════════════════════════════════════════════════
    st.subheader("📸 Final Comparison")
    
    col_orig, col_result = st.columns(2)
    
    with col_orig:
        st.markdown("**Original HDR Image**")
        fig_orig, ax_orig = plt.subplots(figsize=(6, 6))
        hdr_display = np.clip(hdr_image / hdr_image.max(), 0, 1)
        ax_orig.imshow(hdr_display, cmap='gray')
        ax_orig.axis('off')
        st.pyplot(fig_orig)
    
    with col_result:
        output_title = "**Final LDR Output**" if apply_postprocess else "**Tone Mapped Output (No Postprocess)**"
        st.markdown(output_title)
        fig_result, ax_result = plt.subplots(figsize=(6, 6))
        output_display = np.clip(ldr_output / 255.0, 0, 1)
        ax_result.imshow(output_display, cmap='gray')
        ax_result.axis('off')
        st.pyplot(fig_result)
    
    # ════════════════════════════════════════════════════════════════
    # Export & Statistics
    # ════════════════════════════════════════════════════════════════
    st.divider()
    st.subheader("💾 Export & Statistics")
    
    col_stats, col_export = st.columns(2)
    
    with col_stats:
        st.markdown("**Image Statistics**")
        pipeline_info = f"""
        **Pipeline Configuration:**
        - Preprocess: {'✓ Enabled' if apply_preprocess else '✗ Disabled'}
        - Tone Mapping: ✓ Enabled (γ={gamma_tonemap})
        - Postprocess: {'✓ Enabled' if apply_postprocess else '✗ Disabled'}
        
        **Original HDR:**
        - Shape: {hdr_image.shape}
        - Min: {hdr_image.min():.4f}
        - Max: {hdr_image.max():.4f}
        - Mean: {hdr_image.mean():.4f}
        
        **Final Output:**
        - Shape: {ldr_output.shape}
        - Min: {ldr_output.min():.2f}
        - Max: {ldr_output.max():.2f}
        - Mean: {ldr_output.mean():.2f}
        """
        st.code(pipeline_info, language="text")
    
    with col_export:
        st.markdown("**Download Results**")
        
        # Convert to bytes for download
        output_path = "output.png"
        from PIL import Image
        img_pil = Image.fromarray(ldr_output.astype(np.uint8), mode='L')
        
        # Save temporarily
        img_pil.save(output_path)
        
        with open(output_path, "rb") as f:
            st.download_button(
                label="📥 Download Output (PNG)",
                data=f,
                file_name="tone_mapped_output.png",
                mime="image/png"
            )
        
        if auto_save:
            st.success("✓ Output auto-saved to `output.png`")

else:
    st.info("👈 Please select an HDR image from the sidebar to get started")

# ════════════════════════════════════════════════════════════════
# Footer
# ════════════════════════════════════════════════════════════════
st.divider()
st.markdown(
    """
    **Reference:** Liu et al., "An enhancement framework based on gradient domain tone mapping 
    and fuzzy logical for X-ray image of complex workpiece", NDT&E Int. 2021
    """
)
