1007 series, 999 patients

landmark coverage:
  costophrenic_recess_left      1007  100%
  costophrenic_recess_right     1007  100%
  lung_apex_left                1007  100%
  lung_apex_right               1007  100%
  lung_centroid_left            1007  100%
  lung_centroid_right           1007  100%

population relations (mm):
  apex_separation_mm         median    145.4  scale   27.9  p1..p99 [   63.7,  210.6]  flagged   0  -- left minus right apex, in x (+x = patient left)
  recess_separation_mm       median    162.3  scale   33.1  p1..p99 [   76.2,  236.3]  flagged   0  -- left minus right costophrenic recess, in x
  centroid_separation_mm     median    145.1  scale   18.1  p1..p99 [  112.7,  190.1]  flagged   0  -- left minus right lung centroid, in x
  lung_height_left_mm        median    314.0  scale   32.9  p1..p99 [  205.3,  400.3]  flagged  13  -- apex above recess, left lung
  lung_height_right_mm       median    313.7  scale   33.3  p1..p99 [  200.7,  398.8]  flagged  11  -- apex above recess, right lung
  apex_z_asymmetry_mm        median      0.0  scale    4.0  p1..p99 [   -1.5,    8.3]  flagged   0  -- apex height difference L-R
  recess_z_asymmetry_mm      median     -0.0  scale    4.0  p1..p99 [  -10.4,   12.2]  flagged  16  -- recess depth difference L-R (right is usually higher: liver)
  centroid_midline_mm        median     -1.8  scale   12.3  p1..p99 [  -38.6,   30.3]  flagged   4  -- midpoint of the two centroids in x (0 = scanner midline)

sanity on the medians (these are anatomy, not thresholds):
  [OK ] apex separation positive (left apex is at +x)
  [OK ] lung height 150-350mm
  [OK ] centroid midline within 30mm of 0

33 series flagged on at least one relation (3.3%):
  LIDC-IDRI-0403     lung_height_left_mm=165, lung_height_right_mm=164
  LIDC-IDRI-0812     lung_height_left_mm=425, lung_height_right_mm=425
  LIDC-IDRI-0706     lung_height_left_mm=638, lung_height_right_mm=637
  LIDC-IDRI-0793     lung_height_left_mm=411, lung_height_right_mm=411
  LIDC-IDRI-0509     lung_height_left_mm=677, lung_height_right_mm=678
  LIDC-IDRI-0309     lung_height_left_mm=400, lung_height_right_mm=400
  LIDC-IDRI-0792     lung_height_left_mm=408, lung_height_right_mm=407
  LIDC-IDRI-0101     lung_height_left_mm=446, lung_height_right_mm=444
  LIDC-IDRI-0324     lung_height_left_mm=408, lung_height_right_mm=408
  LIDC-IDRI-0136     lung_height_left_mm=455, lung_height_right_mm=455
  LIDC-IDRI-0741     lung_height_left_mm=634, lung_height_right_mm=633
  LIDC-IDRI-0858     lung_height_left_mm=179
  LIDC-IDRI-0958     lung_height_left_mm=401
  LIDC-IDRI-0354     recess_z_asymmetry_mm=16
  LIDC-IDRI-0652     recess_z_asymmetry_mm=27
  LIDC-IDRI-0417     recess_z_asymmetry_mm=-70
  LIDC-IDRI-0806     recess_z_asymmetry_mm=-17
  LIDC-IDRI-0437     recess_z_asymmetry_mm=-21
  LIDC-IDRI-0103     recess_z_asymmetry_mm=17
  LIDC-IDRI-0963     recess_z_asymmetry_mm=20

13 series are anatomically impossible, not merely unusual (1.3%) -- drop these:
  LIDC-IDRI-0812     1.3.6.1.4.1.14519.5.2.1.6279.6001.109203031650348694844047454215
  LIDC-IDRI-0309     1.3.6.1.4.1.14519.5.2.1.6279.6001.199261544234308780356714831537
  LIDC-IDRI-0509     1.3.6.1.4.1.14519.5.2.1.6279.6001.200783211179294046901893379828
  LIDC-IDRI-0136     1.3.6.1.4.1.14519.5.2.1.6279.6001.207201727479884428632451006739
  LIDC-IDRI-0706     1.3.6.1.4.1.14519.5.2.1.6279.6001.213262690132344539993794081003
  LIDC-IDRI-0793     1.3.6.1.4.1.14519.5.2.1.6279.6001.223098610241551815995595311693
  LIDC-IDRI-0741     1.3.6.1.4.1.14519.5.2.1.6279.6001.244995725416880806339898556632
  LIDC-IDRI-0958     1.3.6.1.4.1.14519.5.2.1.6279.6001.281254424459536762125243157973
  LIDC-IDRI-0792     1.3.6.1.4.1.14519.5.2.1.6279.6001.300392272203629213913702120739
  LIDC-IDRI-0858     1.3.6.1.4.1.14519.5.2.1.6279.6001.309955999522338651429118207446
  LIDC-IDRI-0324     1.3.6.1.4.1.14519.5.2.1.6279.6001.334166493392278943610545989413
  LIDC-IDRI-0101     1.3.6.1.4.1.14519.5.2.1.6279.6001.340202188094259402036602717327
  LIDC-IDRI-0403     1.3.6.1.4.1.14519.5.2.1.6279.6001.952265563663939823135367733681

-> /data/lidc/drr_lm/landmark_qc.json

Look at the flagged cases before training. The contamination rate is 3.3%; if that is above a few percent the extraction needs work, not the model.