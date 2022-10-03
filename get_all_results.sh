# script to obtain all figures in the paper
# This script takes as input files:
# exp data
# TCGA data
# TCGA clinical data
# clinical data
# pam50 labels
# bref purities
# RP_2_MLN_pvals_clear_depth6.csv
# /home/bioit/mlarmuse/Documents/Projects/Almac_expression/pam50_TCGA.csv

# perform the tissue level analysis
cd scripts_paper/
python Classification_table.py

# plot Focal adhesion PCA plot
python plot_centroids_on_scatter.py

# perform data analysis of the MLN centroid, also plots TP53 histogram
python UZ_centroid_random_tester.py

# perform the majority voting
python identify_seeding_clone.py

# process the variant snp-pileup data
# python scripts_paper/process_snp_pileup.py

# Compare the two seeding scores (i.e. signature derived vs RNA-seq derived somatic variants
python compare_seeding_scores.py

# Extra test for the significance testing to validate the obtained p-values
python significance_testing_seeds.py

# plot the majority voting
python plot_binary_dfs.py

# plot the Gleason scores
python Figure_Gleason_score_PAM50_seeds.py

# plot the scatter plot with the overrepresented signatures and figure S4
python plot_background_distributions.py

# plot the relative distances to the Centroid
python plot_relative_distances_centroids.py


# Supplementary Figures
# perform PCA plot of the UZ cohort
python doughnut_plot_signatures.py
python cohort_exploration.py
python get_Supplementary_Table1.py
python Description_table_TCGA.py
python analysis_TCGA_PAM50.py

