#BiocManager::install("genefu")

library(genefu)

modality = "FPKM"
path = paste0(c('/home/bioit/mlarmuse/Documents/Data/Almac_project/processed_expression/stringtie_exp_mat_', modality, '.csv.gz'),
              collapse='')
#path = '/home/bioit/mlarmuse/Documents/Data/Almac_project/processed_expression/gene_exp_mat_TPM_genes.csv.gz'
tpm_data = read.csv(path, row.names = 1)

gtf_file = '/home/bioit/mlarmuse/Documents/Data/Almac_project/Nextflow/star_rsem/stringtie/HR_ID11_MLN1.gene.abundance.txt'
gene_annot = read.csv(gtf_file, sep = "\t")

gene_annot <- gene_annot[,c("Gene.ID","Gene.Name")]
names(gene_annot) <- c("Ensemble.ID","Gene.Symbol")

pam50_predictions_all <- molecular.subtyping(sbt.model = "pam50", data = tpm_data, annot = gene_annot, do.mapping = FALSE)

outpath = paste0(c("/home/bioit/mlarmuse/Documents/Projects/Almac_expression/pam50_", modality,".csv"), 
                 collapse='')

write.csv(pam50_predictions_all$subtype.proba, outpath)
data("pam50")
str("pam50")
pam50$centroids
