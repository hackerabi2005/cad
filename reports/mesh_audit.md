# 3D Mesh Audit Report: BodyParts3D Anatomical Heart Model

## 1. Source and Licensing
- **Source**: BodyParts3D / Anatomography (Database Center for Life Science, Japan)
- **License**: Creative Commons Attribution-ShareAlike 2.1 Japan (CC BY-SA 2.1 JP)
- **Attribution**: BodyParts3D, © The Database Center for Life Science licensed under CC Attribution-Share Alike 2.1 Japan
- **Original Path**: `data/raw/BP51782_FMA3_2_1_inference_isa_FMA67135_Postnatal_anatomical_structure/` (Preserved read-only)

## 2. Anatomy & Segment Extraction (Branch A Taken)
Separate high-fidelity coronary artery meshes were identified in the source files, allowing the direct anatomical extraction of:
1. **LAD (Left Anterior Descending)**:
   - Trunk of anterior interventricular branch
   - First, second, third diagonal and septal branches
   - Total faces: 13,574
2. **LCX (Left Circumflex)**:
   - Trunk of circumflex branch
   - First posterior ventricular branch & left marginal artery
   - Total faces: 19,160
3. **RCA (Right Coronary Artery)**:
   - Trunk of right coronary artery
   - Marginal, conus, AV nodal, anterior atrial, and posterior ventricular branches
   - Total faces: 13,764
4. **Ascending Aorta & Bulb**:
   - Total faces: 6,572
5. **Heart Chambers (Myocardium Shell)**:
   - Left and right ventricular free walls, interventricular septum, atrial walls, and auricles
   - Total faces after quadric decimation: 30,530

## 3. Performance Metrics
- **Output File**: `web/public/models/heart.glb`
- **File Size**: 1.67 MB
- **Total Triangles**: 83,600 triangles (Within strict performance budget of ≤ 100,000 triangles)
- **Framerate Target**: 60 FPS in modern WebGL / Three.js; benchmarked at 49.3 ms mean frame time (~20.3 FPS) under pure CPU software rendering (`--disable-gpu`)
- **Named Nodes Verified**: `LAD`, `LCX`, `RCA`, `Aorta`, `Heart_Muscle`
