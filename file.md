007 series, 999 patients

landmark coverage:
  costophrenic_recess_left      1007  100%
  costophrenic_recess_right     1007  100%
  lung_apex_left                1007  100%
  lung_apex_right               1007  100%
  lung_centroid_left            1007  100%
  lung_centroid_right           1007  100%

population relations (mm):
  apex_separation_mm         median    145.4  MAD   27.9  p1..p99 [   63.7,  210.6]  flagged   0  -- left minus right apex, in x (+x = patient left)
  recess_separation_mm       median    162.3  MAD   33.1  p1..p99 [   76.2,  236.3]  flagged   0  -- left minus right costophrenic recess, in x
  centroid_separation_mm     median    145.1  MAD   17.7  p1..p99 [  112.7,  190.1]  flagged   0  -- left minus right lung centroid, in x
  lung_height_left_mm        median    314.0  MAD   32.9  p1..p99 [  205.3,  400.3]  flagged   7  -- apex above recess, left lung
  lung_height_right_mm       median    313.7  MAD   33.3  p1..p99 [  200.7,  398.8]  flagged   5  -- apex above recess, right lung
  apex_z_asymmetry_mm        median      0.0  MAD    0.0  p1..p99 [   -1.5,    8.3]  flagged 413  -- apex height difference L-R
  recess_z_asymmetry_mm      median     -0.0  MAD    0.4  p1..p99 [  -10.4,   12.2]  flagged 122  -- recess depth difference L-R (right is usually higher: liver)
  centroid_midline_mm        median     -1.8  MAD   12.2  p1..p99 [  -38.6,   30.3]  flagged   4  -- midpoint of the two centroids in x (0 = scanner midline)

sanity on the medians (these are anatomy, not thresholds):
  [OK ] apex separation positive (left apex is at +x)
  [OK ] lung height 150-350mm
  [OK ] centroid midline within 30mm of 0

444 series flagged on at least one relation (44.1%):
  LIDC-IDRI-0403     lung_height_left_mm=165, lung_height_right_mm=164, apex_z_asymmetry_mm=0
  LIDC-IDRI-0136     lung_height_left_mm=455, lung_height_right_mm=455, apex_z_asymmetry_mm=-1
  LIDC-IDRI-0706     lung_height_left_mm=638, lung_height_right_mm=637
  LIDC-IDRI-0858     lung_height_left_mm=179, apex_z_asymmetry_mm=-2
  LIDC-IDRI-0509     lung_height_left_mm=677, lung_height_right_mm=678
  LIDC-IDRI-0101     lung_height_left_mm=446, apex_z_asymmetry_mm=1
  LIDC-IDRI-0741     lung_height_left_mm=634, lung_height_right_mm=633
  LIDC-IDRI-0801     apex_z_asymmetry_mm=3, recess_z_asymmetry_mm=6
  LIDC-IDRI-0953     apex_z_asymmetry_mm=12, recess_z_asymmetry_mm=6
  LIDC-IDRI-0915     apex_z_asymmetry_mm=1, recess_z_asymmetry_mm=-4
  LIDC-IDRI-0439     apex_z_asymmetry_mm=6, recess_z_asymmetry_mm=-4
  LIDC-IDRI-0781     apex_z_asymmetry_mm=10, recess_z_asymmetry_mm=3
  LIDC-IDRI-0371     apex_z_asymmetry_mm=7, recess_z_asymmetry_mm=-11
  LIDC-IDRI-0955     apex_z_asymmetry_mm=7, recess_z_asymmetry_mm=10
  LIDC-IDRI-0635     apex_z_asymmetry_mm=6, recess_z_asymmetry_mm=2
  LIDC-IDRI-0620     apex_z_asymmetry_mm=8, recess_z_asymmetry_mm=8
  LIDC-IDRI-1008     apex_z_asymmetry_mm=1, recess_z_asymmetry_mm=-3
  LIDC-IDRI-0354     apex_z_asymmetry_mm=13, recess_z_asymmetry_mm=16
  LIDC-IDRI-0688     apex_z_asymmetry_mm=7, recess_z_asymmetry_mm=-4
  LIDC-IDRI-0550     apex_z_asymmetry_mm=-5, recess_z_asymmetry_mm=-8

-> /data/lidc/drr_lm/landmark_qc.json
-> /data/lidc/qc.png

Look at the flagged cases before training. The contamination rate is 44.1%; if that is above a few percent the extraction needs work, not the model.