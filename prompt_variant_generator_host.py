#!/usr/bin/env python3

import streamlit as st
import pandas as pd
import replicate
import base64
import requests
import json
import time
from pathlib import Path
import zipfile
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import logging
from io import BytesIO
import mimetypes
import ssl
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import urllib3
import traceback
import uuid  # For X-TT-LOGID header

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# API Configuration from Streamlit Secrets
ARK_API_KEY = st.secrets["ARK_API_KEY"]
REPLICATE_API_TOKEN = st.secrets["REPLICATE_API_TOKEN"]

# Set Replicate API token
replicate.api_token = REPLICATE_API_TOKEN

API_URL = "https://ark.ap-southeast.bytepluses.com/api/v3/images/generations"
MODEL_NAME = "seedream-4-0-250828"  # Seedream 4.0 model

# Image quality settings
OUTPUT_SIZE = "2304x4096"  # 4K resolution

# Parallel processing configuration
MAX_WORKERS_PROMPTS = 5  # For GPT-5 prompt generation
MAX_WORKERS_IMAGES = 10  # For Seedream image generation

# Prompt template for variant generation
VARIANT_PROMPT_TEMPLATE = """Create four distinct but thematically related variants of the base image described below. Each variant should feel like a different moment from the same person's life, maintaining overall aesthetic consistency while introducing significant visual variety.

Gender: {gender}
Base Scene Prompt: {base_prompt}

REQUIRED CHANGES for each variant:
1. **OUTFIT**: Completely different outfit, but tonally appropriate (e.g., if base is casual streetwear, try: leather jacket, oversized sweater, denim jacket, or button-down shirt - different items, same vibe)

2. **BACKGROUND**: Different location within the same world (e.g., if base is Tokyo street → variants could be: subway platform, rooftop terrace, quiet alley, convenience store interior, park bench, pedestrian crossing)

3. **POSE**: Natural variation in body language (e.g., walking → standing, sitting, leaning against wall, looking over shoulder, hands in pockets, holding coffee cup)

4. **MAINTAIN**: Same lighting mood (golden hour/overcast/night), same photographic style (cinematic/editorial), same cultural aesthetic (Japanese minimalism), realistic and natural, no crowd.

Do NOT repeat the same outfit or background across variants. Each should be visually distinct while feeling connected thematically.

Output Goal: Generate 4 distinct prompts (Variant 1–4). Each must be a complete, standalone image prompt including outfit details, specific background location, pose description, and lighting/style notes. {gender} should be looking at the camera.

Format your response as:
Variant 1: [full detailed prompt with specific outfit, background, and pose]
Variant 2: [full detailed prompt with specific outfit, background, and pose]
Variant 3: [full detailed prompt with specific outfit, background, and pose]
Variant 4: [full detailed prompt with specific outfit, background, and pose]"""

class SSLAdapter(HTTPAdapter):
    """Custom SSL adapter to handle SSL connection issues"""
    def init_poolmanager(self, *args, **kwargs):
        context = ssl.create_default_context()
        context.set_ciphers('DEFAULT@SECLEVEL=1')
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
        kwargs['ssl_context'] = context
        return super().init_poolmanager(*args, **kwargs)


class PromptVariantGenerator:
    """Handles GPT-5 based prompt variant generation"""
    
    def __init__(self):
        self.local = threading.local()
    
    def generate_variants_with_retry(self, base_prompt, gender, max_retries=3):
        """Generate 4 variants of the base prompt using GPT-5 via Replicate with retry logic"""
        
        prompt = VARIANT_PROMPT_TEMPLATE.format(
            gender=gender,
            base_prompt=base_prompt
        )
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Generating variants for prompt (attempt {attempt + 1}/{max_retries})")
                
                # Stream output from GPT-5
                output_text = ""
                for event in replicate.stream(
                    "openai/gpt-5",
                    input={
                        "prompt": prompt,
                        "messages": [],
                        "verbosity": "medium",
                        "image_input": [],
                        "reasoning_effort": "minimal"
                    }
                ):
                    output_text += str(event)
                
                # Parse variants from output
                variants = self.parse_variants(output_text)
                
                if len(variants) == 4:
                    logger.info(f"Successfully generated 4 variants")
                    return variants
                else:
                    logger.warning(f"Expected 4 variants, got {len(variants)}. Retrying...")
                    if attempt < max_retries - 1:
                        time.sleep(2)
                        continue
                    else:
                        # Return what we have or fallback
                        return variants if variants else [base_prompt] * 4
                        
            except Exception as e:
                logger.error(f"Error generating variants (attempt {attempt + 1}): {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(3)
                    continue
                else:
                    # Fallback to base prompt if all retries fail
                    logger.warning("All retries failed, using base prompt as variants")
                    return [base_prompt] * 4
        
        return [base_prompt] * 4
    
    def parse_variants(self, output_text):
        """Parse variant prompts from GPT-5 output"""
        variants = []
        lines = output_text.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            # Look for "Variant N:" pattern
            if line.lower().startswith('variant'):
                # Extract the prompt after the colon
                if ':' in line:
                    variant_text = line.split(':', 1)[1].strip()
                    if variant_text:
                        variants.append(variant_text)
        
        logger.info(f"Parsed {len(variants)} variants from output")
        return variants


class SeedreamImageGenerator:
    """Handles Seedream image generation with face-swap using Ark API"""
    
    def __init__(self):
        self.local = threading.local()
    
    def get_session(self):
        """Get or create a thread-local session"""
        if not hasattr(self.local, 'session'):
            session = requests.Session()
            
            retry_strategy = Retry(
                total=3,
                backoff_factor=1,
                status_forcelist=[429, 500, 502, 503, 504],
                allowed_methods=["HEAD", "GET", "POST"]
            )
            
            ssl_adapter = SSLAdapter(max_retries=retry_strategy)
            session.mount("https://", ssl_adapter)
            session.mount("http://", HTTPAdapter(max_retries=retry_strategy))
            
            session.headers.update({
                "User-Agent": "SeedrameImageGenerator/1.0",
                "Accept": "application/json",
                "Connection": "keep-alive"
            })
            
            session.verify = False
            self.local.session = session
            
        return self.local.session
    
    def encode_image_to_base64(self, image_bytes):
        """Convert image bytes to base64 string with data URI format"""
        try:
            base64_string = base64.b64encode(image_bytes).decode('utf-8')
            # Default to JPEG for uploaded images
            return f"data:image/jpeg;base64,{base64_string}"
        except Exception as e:
            logger.error(f"Error encoding image: {str(e)}")
            return None
    
    def generate_image_with_faceswap(self, prompt, selfie_base64, max_retries=3):
        """Generate image using Seedream with face-swap using Ark API v3"""
        
        session = self.get_session()
        
        # Prepare payload matching Ark API v3 format
        payload = {
            "model": MODEL_NAME,
            "prompt": prompt,
            "image": [selfie_base64],
            "response_format": "url",
            "size": OUTPUT_SIZE,
            "watermark": False,
            "stream": False
        }
        
        # Headers matching Ark API requirements
        headers = {
            "X-TT-LOGID": str(uuid.uuid4()),
            "Content-Type": "application/json",
            "Authorization": f"Bearer {ARK_API_KEY}"
        }
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Generating image (attempt {attempt + 1}/{max_retries})")
                
                response = session.post(
                    API_URL,
                    headers=headers,
                    json=payload,
                    timeout=120,
                    verify=False
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Parse response matching Ark API v3 format
                    if 'data' in result and isinstance(result['data'], list) and len(result['data']) > 0:
                        image_data = result['data'][0]
                        image_url = image_data.get('url')
                        
                        if image_url:
                            logger.debug(f"Got image URL: {image_url[:50]}...")
                            
                            # Download the image
                            image_response = session.get(
                                image_url,
                                timeout=60,
                                verify=False
                            )
                            
                            if image_response.status_code == 200:
                                logger.info("Successfully generated and downloaded image")
                                return image_response.content
                            else:
                                logger.error(f"Failed to download image: {image_response.status_code}")
                        else:
                            logger.error(f"No URL in response: {result}")
                    else:
                        logger.error(f"Unexpected response format: {result}")
                    
                    logger.warning(f"Generation failed or no images returned")
                
                else:
                    logger.error(f"API request failed with status {response.status_code}: {response.text[:200]}")
                
                # Retry logic
                if attempt < max_retries - 1:
                    wait_time = 3 * (attempt + 1)
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                
            except requests.exceptions.Timeout:
                logger.error(f"Request timeout (attempt {attempt + 1})")
                if attempt < max_retries - 1:
                    time.sleep(5)
                    continue
                    
            except Exception as e:
                logger.error(f"Error generating image: {str(e)}")
                logger.error(traceback.format_exc())
                if attempt < max_retries - 1:
                    time.sleep(3)
                    continue
        
        return None


def process_single_prompt_row(row_data, variant_generator):
    """Process a single prompt row to generate variants"""
    prompt_idx, base_prompt, gender = row_data
    
    try:
        variants = variant_generator.generate_variants_with_retry(base_prompt, gender)
        logger.info(f"Completed variant generation for prompt {prompt_idx + 1}")
        return (prompt_idx, base_prompt, gender, variants)
    except Exception as e:
        logger.error(f"Error processing prompt {prompt_idx + 1}: {str(e)}")
        return (prompt_idx, base_prompt, gender, [base_prompt] * 4)


def process_single_image(task, image_generator, selfie_base64):
    """Process a single image generation task"""
    prompt_idx, variant_idx, variant_prompt, base_prompt = task
    
    try:
        image_bytes = image_generator.generate_image_with_faceswap(
            variant_prompt,
            selfie_base64
        )
        
        if image_bytes:
            logger.info(f"Successfully generated image for prompt {prompt_idx + 1}, variant {variant_idx + 1}")
            return (prompt_idx, variant_idx, variant_prompt, base_prompt, image_bytes, True)
        else:
            logger.warning(f"Failed to generate image for prompt {prompt_idx + 1}, variant {variant_idx + 1}")
            return (prompt_idx, variant_idx, variant_prompt, base_prompt, None, False)
            
    except Exception as e:
        logger.error(f"Error generating image for prompt {prompt_idx + 1}, variant {variant_idx + 1}: {str(e)}")
        return (prompt_idx, variant_idx, variant_prompt, base_prompt, None, False)


def create_download_package(image_results, metadata_df):
    """Create a ZIP file containing all images and metadata"""
    zip_buffer = BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # Add metadata CSV
        csv_buffer = BytesIO()
        metadata_df.to_csv(csv_buffer, index=False)
        zip_file.writestr('metadata.csv', csv_buffer.getvalue())
        
        # Add images
        for prompt_idx, variant_idx, variant_prompt, base_prompt, image_bytes, success in image_results:
            if success and image_bytes:
                filename = f"prompt_{prompt_idx + 1:03d}_variant_{variant_idx + 1}.jpg"
                zip_file.writestr(filename, image_bytes)
    
    zip_buffer.seek(0)
    return zip_buffer


def main():
    st.set_page_config(
        page_title="Prompt Variant Generator",
        page_icon="🎨",
        layout="wide"
    )
    
    st.title("🎨 Prompt Variant Image Generator")
    st.markdown("Generate multiple image variants from base prompts using GPT-5 and Seedream")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        max_workers_prompts = st.slider(
            "Prompt Workers",
            min_value=1,
            max_value=10,
            value=MAX_WORKERS_PROMPTS,
            help="Number of parallel workers for prompt generation"
        )
        max_workers_images = st.slider(
            "Image Workers",
            min_value=1,
            max_value=20,
            value=MAX_WORKERS_IMAGES,
            help="Number of parallel workers for image generation"
        )
        
        st.markdown("---")
        st.markdown("""
        ### How it works:
        1. **Upload CSV**: with `prompt` and `gender` columns
        2. **Upload Selfie**: face to use for all images
        3. **Generate**: Creates 4 variants per prompt
        4. **Download**: Get ZIP with images + metadata
        """)
    
    # Main content
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📄 Upload CSV")
        csv_file = st.file_uploader(
            "Choose CSV file",
            type=['csv'],
            help="CSV must have 'prompt' and 'gender' columns"
        )
        
        if csv_file:
            try:
                df_preview = pd.read_csv(csv_file, encoding='utf-8-sig')
                df_preview.columns = df_preview.columns.str.strip()
                st.dataframe(df_preview.head(), use_container_width=True)
                st.info(f"📊 Found {len(df_preview)} prompts")
            except Exception as e:
                st.error(f"Error reading CSV: {str(e)}")
    
    with col2:
        st.subheader("📸 Upload Selfie")
        selfie_file = st.file_uploader(
            "Choose selfie image",
            type=['jpg', 'jpeg', 'png', 'webp'],
            help="This face will be used for all generated images"
        )
        
        if selfie_file:
            st.image(selfie_file, caption="Uploaded Selfie", width=300)
    
    # Process button
    st.markdown("---")
    
    if st.button("🚀 Generate Images", type="primary", use_container_width=True):
        
        if not csv_file or not selfie_file:
            st.error("❌ Please upload both CSV file and selfie image")
            return
        
        try:
            # Read CSV with proper encoding to handle BOM
            csv_file.seek(0)
            df = pd.read_csv(csv_file, encoding='utf-8-sig')
            df.columns = df.columns.str.strip()
            
            if 'prompt' not in df.columns or 'gender' not in df.columns:
                st.error("❌ CSV must have 'prompt' and 'gender' columns")
                return
            
            # Encode selfie
            selfie_bytes = selfie_file.read()
            image_generator = SeedreamImageGenerator()
            selfie_base64 = image_generator.encode_image_to_base64(selfie_bytes)
            
            if not selfie_base64:
                st.error("❌ Failed to encode selfie image")
                return
            
            # Initialize generators
            variant_generator = PromptVariantGenerator()
            
            # Prepare data
            total_prompts = len(df)
            prompt_rows = [
                (idx, row['prompt'], row['gender'])
                for idx, row in df.iterrows()
            ]
            
            st.info(f"🔄 Processing {total_prompts} prompts...")
            
            # STEP 1: Generate prompt variants (parallel)
            st.markdown("### Step 1: Generating Prompt Variants")
            prompt_progress = st.progress(0)
            prompt_status = st.empty()
            
            prompt_results = []
            
            with ThreadPoolExecutor(max_workers=max_workers_prompts) as executor:
                futures = [
                    executor.submit(
                        process_single_prompt_row,
                        row_data,
                        variant_generator
                    )
                    for row_data in prompt_rows
                ]
                
                for i, future in enumerate(as_completed(futures)):
                    try:
                        result = future.result()
                        prompt_results.append(result)
                        
                        # Update progress in main thread (safe!)
                        progress = (i + 1) / total_prompts
                        prompt_progress.progress(progress)
                        prompt_status.text(f"Generated variants for {i + 1}/{total_prompts} prompts")
                        
                    except Exception as e:
                        logger.error(f"Error in prompt generation: {str(e)}")
                        prompt_status.text(f"Error processing prompt: {str(e)}")
            
            st.success(f"✓ Generated {len(prompt_results)} prompt sets")
            
            # STEP 2: Generate images (parallel)
            st.markdown("### Step 2: Generating Images")
            
            # Prepare image generation tasks
            image_tasks = []
            for prompt_idx, base_prompt, gender, variants in prompt_results:
                for variant_idx, variant_prompt in enumerate(variants):
                    image_tasks.append((prompt_idx, variant_idx, variant_prompt, base_prompt))
            
            total_images = len(image_tasks)
            image_progress = st.progress(0)
            image_status = st.empty()
            
            image_results = []
            success_count = 0
            failed_count = 0
            
            with ThreadPoolExecutor(max_workers=max_workers_images) as executor:
                futures = [
                    executor.submit(
                        process_single_image,
                        task,
                        image_generator,
                        selfie_base64
                    )
                    for task in image_tasks
                ]
                
                for i, future in enumerate(as_completed(futures)):
                    try:
                        result = future.result()
                        image_results.append(result)
                        
                        # Track success/failure
                        if result[5]:  # success flag
                            success_count += 1
                        else:
                            failed_count += 1
                        
                        # Update progress in main thread (safe!)
                        progress = (i + 1) / total_images
                        image_progress.progress(progress)
                        image_status.text(
                            f"Generated {i + 1}/{total_images} images "
                            f"(Success: {success_count}, Failed: {failed_count})"
                        )
                        
                    except Exception as e:
                        logger.error(f"Error in image generation: {str(e)}")
                        failed_count += 1
            
            st.success(
                f"✓ Generated {success_count} images successfully, "
                f"{failed_count} failed"
            )
            
            # STEP 3: Create metadata and download package
            st.markdown("### Step 3: Preparing Download Package")
            
            # Create metadata DataFrame
            metadata_records = []
            for prompt_idx, variant_idx, variant_prompt, base_prompt, image_bytes, success in image_results:
                metadata_records.append({
                    'prompt_id': prompt_idx + 1,
                    'variant_id': variant_idx + 1,
                    'base_prompt': base_prompt,
                    'variant_prompt': variant_prompt,
                    'image_filename': f"prompt_{prompt_idx + 1:03d}_variant_{variant_idx + 1}.jpg" if success else "FAILED",
                    'status': 'success' if success else 'failed'
                })
            
            metadata_df = pd.DataFrame(metadata_records)
            
            # Show metadata preview
            st.dataframe(metadata_df, use_container_width=True)
            
            # Create ZIP package
            with st.spinner("Creating download package..."):
                zip_buffer = create_download_package(image_results, metadata_df)
            
            # Download button
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            st.download_button(
                label="⬇️ Download All Images & Metadata",
                data=zip_buffer,
                file_name=f"generated_images_{timestamp}.zip",
                mime="application/zip",
                use_container_width=True
            )
            
            st.success("🎉 Processing complete! Click the button above to download.")
            
            # Show summary statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Prompts", total_prompts)
            with col2:
                st.metric("Total Images", total_images)
            with col3:
                st.metric("Successful", success_count)
            with col4:
                st.metric("Failed", failed_count)
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
