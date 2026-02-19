import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

# ---------------------------------------------------------
# 1-5. PREVIOUS ALGORITHMS (Simplified for space)
# ---------------------------------------------------------
def apply_clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    cl = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)

def apply_msrcr(img, sigmas=[15, 80, 250]):
    img_float = np.float64(img) + 1.0 
    retinex = sum(np.log10(img_float) - np.log10(cv2.GaussianBlur(img_float, (0, 0), s)) for s in sigmas) / len(sigmas)
    return np.uint8(cv2.normalize(retinex, None, 0, 255, cv2.NORM_MINMAX))

# ---------------------------------------------------------
# 3. Fusion Method (Simplified Ancuti) - FIXED FOR NUMPY
# ---------------------------------------------------------
def apply_fusion(img):
    # Convert to LAB and cast to float32 to prevent numpy casting errors
    result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    
    a_avg, b_avg = np.average(result[:, :, 1]), np.average(result[:, :, 2])
    
    # Use standard assignment instead of in-place subtraction (-=)
    result[:, :, 1] = result[:, :, 1] - ((a_avg - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((b_avg - 128) * (result[:, :, 0] / 255.0) * 1.1)
    
    # Clip the values to valid ranges and convert back to uint8
    result = np.clip(result, 0, 255).astype(np.uint8)
    img1 = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    
    img2 = apply_clahe(img1)
    
    # Calculate Weight Maps
    w1 = np.abs(cv2.Laplacian(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), cv2.CV_64F)) + 1e-5
    w2 = np.abs(cv2.Laplacian(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY), cv2.CV_64F)) + 1e-5
    
    W1, W2 = w1 / (w1 + w2), w2 / (w1 + w2)
    
    fused = np.zeros_like(img, dtype=np.float64)
    for i in range(3): 
        fused[:, :, i] = img1[:, :, i] * W1 + img2[:, :, i] * W2
        
    return np.uint8(np.clip(fused, 0, 255))

def apply_agc(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    gamma = np.log(0.5) / np.log(np.mean(v) / 255.0) 
    table = np.array([((i / 255.0) ** (1.0/gamma)) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.cvtColor(cv2.merge((h, s, cv2.LUT(v, table))), cv2.COLOR_HSV2BGR)

def apply_usm_bilateral(img):
    smoothed = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
    unsharp = cv2.addWeighted(smoothed, 1.5, cv2.GaussianBlur(smoothed, (0, 0), 3.0), -0.5, 0)
    return cv2.normalize(unsharp, None, 0, 255, cv2.NORM_MINMAX)

# ---------------------------------------------------------
# 6. NEW: Underwater Dark Channel Prior (UDCP)
# ---------------------------------------------------------
def apply_udcp(img):
    # UDCP calculates dark channel only from Green and Blue channels
    b, g, r = cv2.split(img.astype(np.float32) / 255.0)
    dc = cv2.min(b, g) 
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    dark_channel = cv2.erode(dc, kernel)
    
    # Estimate global atmospheric light (A)
    num_brightest = int(max(math.floor(dark_channel.size / 1000), 1))
    indices = np.argsort(dark_channel.ravel())[-num_brightest:]
    A = np.maximum(np.mean((img.reshape(-1, 3) / 255.0)[indices], axis=0), 0.001)
    
    # Estimate and refine transmission map
    norm_dc = cv2.min(b / A[0], g / A[1])
    transmission = np.maximum(cv2.GaussianBlur(1 - 0.95 * cv2.erode(norm_dc, kernel), (15, 15), 0), 0.1)
    
    # Reverse the physical scattering model
    J = np.zeros_like(img, dtype=np.float32)
    for i in range(3):
        J[:,:,i] = (img[:,:,i]/255.0 - A[i]) / transmission + A[i]
        
    return np.uint8(np.clip(cv2.normalize(J, None, 0, 255, cv2.NORM_MINMAX), 0, 255))

# ---------------------------------------------------------
# 7. NEW: Red Channel Compensation + Gray World
# ---------------------------------------------------------
def apply_rcc_wb(img):
    b, g, r = cv2.split(np.float32(img))
    
    # Compensate red channel proportionally to the difference between green and red
    r_compensated = r + 0.5 * (np.mean(g) - np.mean(r)) * (1 - r/255.0) 
    img_comp = cv2.merge((b, g, r_compensated))
    
    # Apply Gray World White Balancing
    avg_color = np.mean(img_comp, axis=(0, 1))
    scale = np.mean(avg_color) / avg_color
    
    return np.clip(img_comp * scale, 0, 255).astype(np.uint8)

# ---------------------------------------------------------
# 8. NEW: White Patch Retinex (Max-RGB)
# ---------------------------------------------------------
def apply_white_patch(img):
    b, g, r = cv2.split(img.astype(np.float32))
    
    # Scale each channel by its maximum value
    b = np.clip(b * (255.0 / np.max(b)), 0, 255)
    g = np.clip(g * (255.0 / np.max(g)), 0, 255)
    r = np.clip(r * (255.0 / np.max(r)), 0, 255)
    
    return cv2.merge((b, g, r)).astype(np.uint8)

# =========================================================
# Execution and Plotting
# =========================================================
# =========================================================
# Execution and Plotting
# =========================================================
def main():
    image_path = 'image5.jpg'
    original = cv2.imread(image_path)
    
    if original is None:
        print(f"Error: Could not find '{image_path}'.")
        return

    print("Running 8 enhancement algorithms... Please wait.")
    
    results = {
        "0. Original": original,
        "1. CLAHE": apply_clahe(original),
        "2. Multi-Scale Retinex": apply_msrcr(original),
        "3. Fusion Method": apply_fusion(original),
        "4. Adaptive Gamma": apply_agc(original),
        "5. USM + Bilateral": apply_usm_bilateral(original),
        "6. UDCP (Physics Model)": apply_udcp(original),
        "7. Red Comp + Gray World": apply_rcc_wb(original),
        "8. White Patch (Max-RGB)": apply_white_patch(original)
    }

    # FIX 1: Reduced figsize from (18, 12) to (12, 8) to fit on standard screens
    fig, axes = plt.subplots(3, 3, figsize=(12, 8))
    fig.canvas.manager.set_window_title('Ultimate Underwater Enhancement')
    axes = axes.ravel()

    for idx, (title, img_bgr) in enumerate(results.items()):
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        axes[idx].imshow(img_rgb)
        axes[idx].set_title(title, fontsize=10, fontweight='bold') # Reduced font size slightly
        axes[idx].axis('off')

    plt.tight_layout()
    
    # FIX 2: Save a high-resolution copy directly to your folder
    output_filename = "comparison_grid.jpg"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Done! High-res copy saved as '{output_filename}'. Opening graph...")
    
    plt.show()

if __name__ == "__main__":
    main()