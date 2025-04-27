
# Blender Addon Development Summary: RMF Implementation

## 1. Initial Goal & Problem Encountered

The primary goal was to develop a Blender addon (`PieceGen`) that deforms a mesh (initially a cylinder) along a user-editable Bézier curve, including support for controlling the "tilt" or twist around the curve's tangent.

The initial implementation used a function (`evaluate_bezier_spline_with_tilt`) to calculate the curve's position, tangent, and a tilt-aware normal vector at any given parameter $t$. However, visualizations revealed a significant problem: the calculated orientation frames (Tangent, Normal, Binormal) were unstable and exhibited sudden **180-degree flips**, particularly in curved sections or at segment boundaries.

*(Conceptual illustration of the problem: Imagine driving a car along the curve, the direction (Tangent) is fine, but the "up" direction of the car (Normal) suddenly flips upside down at corners).*

## 2. Diagnosis and Bug Fixes

Through analysis and experimentation, we identified several contributing factors:

* **Root Cause - Frame Calculation Instability:** The core issue was traced to the `evaluate_bezier_spline_with_tilt` function's method for calculating the initial normal vector before applying tilt. It relied on calculating cross products between the curve tangent ($\vec{T}$) and a fixed world-space axis ($\vec{RefUp}$, usually World Z or Y).
    * **Instability near Alignment:** When $\vec{T}$ became nearly parallel to $\vec{RefUp}$, the cross product $\vec{T} \times \vec{RefUp}$ became numerically unstable, leading to inconsistent orientation.
    * **Reference Vector Switching:** A hard threshold (`if abs(tangent.dot(WorldZ)) > 0.9999:`) caused the $\vec{RefUp}$ vector to abruptly switch between World Z and World Y. When $\vec{T}$ crossed this angular threshold (often precisely at segment boundaries), $\vec{RefUp}$ would jump relative to $\vec{T}$, causing the resulting cross product ($\vec{B}_{\text{initial}} = \vec{T} \times \vec{RefUp}$) and the subsequent normal ($\vec{N}_{\text{initial}} = \vec{T} \times \vec{B}_{\text{initial}}$) to flip direction.

* **Curve Creation & Appearance:**
    * We initially suspected the curve generation might be faulty. It was found that `curve_data.resolution_u` was set too low (2), making the curve *appear* like a polyline. This was corrected in the generation operator `OBJECT_OT_generate_cylinder_with_curve` for better visual feedback, though it wasn't the direct cause of the flipping.
    * The initial use of `'ALIGNED'` handles might have contributed indirectly by making it easier to create sharp bends near control points during editing, which then triggered the frame calculation instability. This was changed to `'AUTO'` handles in the generation code.

* **World Space Transformation:**
    * We identified that transforming tangent vectors ($\vec{T}$) requires the standard matrix ($M$), while normal ($\vec{N}$) and binormal ($\vec{B}$) vectors require the inverse-transpose ($M^{-T}$) for correctness under non-uniform scaling.
    * The visualization code initially used $M$ for all vectors. This was corrected to use $M^{-T}$ for $\vec{N}$ and $\vec{B}$.
    * It was observed that even with correct transformation and uniform object scale, the *world-space* vectors $(\vec{T}_{world}, \vec{N}_{world}, \vec{B}_{world})$ might not remain perfectly orthogonal due to potential inherited non-uniform scale (e.g., from parenting). A re-orthogonalization step was added to the *visualization* code (`VISUALIZE_OT_curve_frames`) to ensure the displayed markers are perpendicular.

* **Visualization Details:**
    * The gradient material initially appeared solid white; this was resolved by ensuring Blender's **Material Preview** or **Rendered** viewport shading mode was active.
    * The gradient appearance was adjusted (reducing the amount of white) by modifying the `ColorRamp` node stops in the `create_gradient_material` function.

## 3. Solution: Rotation Minimizing Frame (RMF) Implementation

To fundamentally solve the frame instability and flipping, the simple cross-product-based frame calculation was replaced with a robust **Rotation Minimizing Frame (RMF)** algorithm using the **Double Reflection Method**.

* **Code Refactoring:** The complex geometric calculations were moved from `__init__.py` into a dedicated utility module, `rmf_utils.py`.
* **RMF Algorithm (`rmf_utils.calculate_rmf_frames`):**
    * This function calculates a sequence of frames along the curve iteratively.
    * It samples curve positions $\vec{P}_i$ and tangents $\vec{T}_i$ at discrete steps.
    * Starting with a robustly calculated initial frame $(\vec{T}_0, \vec{N}_0)$, it computes the next normal $\vec{N}_{i+1}$ from the previous normal $\vec{N}_i$ using two reflections:
        1.  Reflect $\vec{N}_i$ across the plane perpendicular to the segment vector $\vec{v}_1 = \vec{P}_{i+1} - \vec{P}_i$:
            $$\vec{N}_L = \vec{N}_i - \frac{2 (\vec{v}_1 \cdot \vec{N}_i)}{\|\vec{v}_1\|^2} \vec{v}_1$$
        2.  Reflect the intermediate $\vec{N}_L$ across the plane perpendicular to the average tangent direction $\vec{v}_2 = \vec{T}_i + \vec{T}_{i+1}$:
            $$\vec{N}_{i+1} = \vec{N}_L - \frac{2 (\vec{v}_2 \cdot \vec{N}_L)}{\|\vec{v}_2\|^2} \vec{v}_2$$
    * The resulting sequence of frames $(\vec{P}_k, \vec{T}_k, \vec{N}_k)$ minimizes rotation around the tangent $\vec{T}_k$.

* **Integration (`__init__.py`):**
    * The `deform_mesh_along_curve` function now calls `rmf_utils.calculate_rmf_frames` once to pre-compute the stable RMF frames.
    * For each mesh vertex, it finds the **nearest** pre-calculated RMF frame. *(Note: This is a simplification; higher accuracy would require geometric interpolation (e.g., Slerp) between the two frames bracketing $t_{\text{vertex}}$).*
    * It retrieves the RMF tangent $\vec{T}_{\text{RMF}}$ and normal $\vec{N}_{\text{RMF}}$ from that frame.
    * It calculates the required tilt angle $\theta(t)$ using `rmf_utils.get_interpolated_tilt`.
    * It applies the tilt by rotating $\vec{N}_{\text{RMF}}$ around $\vec{T}_{\text{RMF}}$ to get the final $\vec{N}_{\text{final}}$.
    * It constructs the final frame and transforms the vertex.
    * The `VISUALIZE_OT_curve_frames` operator was similarly updated to calculate and display the tilted RMF frames.

*(Conceptual illustration of the solution: The car driving along the curve now keeps its "up" direction stable, smoothly rotating only as needed to follow the curve's path without sudden flips).*

## 4. Conclusion

The addon now employs a mathematically robust method (RMF via Double Reflection) for calculating curve frames, addressing the core instability and flipping issues of the previous approach. The code has also been refactored for better organization, and several smaller bugs related to visualization and transformation have been corrected. The result should be a much more stable and predictable deformation along the guide curve.
