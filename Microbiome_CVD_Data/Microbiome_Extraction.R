# ────────────────────────────────────────────────────────────────────────────────
# JieZ_2017 – full microbial feature table with CVD label
# ────────────────────────────────────────────────────────────────────────────────
# 1. Load curatedMetagenomicData and pull the relative‑abundance object
# 2. Extract every microbial‑feature column from assay()
# 3. Append key metadata (BMI → bmi, age_category → age_group, disease)
# 4. Create binary has_CVD label (1 = CVD / atherosclerosis, 0 = everything else)
# 5. Write clean ML‑ready CSV (no row names) → jie_2017_cvd_microbiome_full.csv
# ────────────────────────────────────────────────────────────────────────────────

# Install once if needed
BiocManager::install("curatedMetagenomicData")

library(curatedMetagenomicData)
library(SummarizedExperiment)
library(stringr)   # for str_detect
library(dplyr)     # tidy data wrangling
library(tibble)    # rownames_to_column

# 1️⃣  Pull the dataset ----------------------------------------------------------
se <- curatedMetagenomicData("JieZ_2017.relative_abundance", dryrun = FALSE)[[1]]

# 2️⃣  Extract the abundance matrix & convert to data frame ----------------------
abund_df <- assay(se) %>%            # matrix [features × samples]
  t() %>%                            # transpose → samples × features
  as.data.frame() %>%
  rownames_to_column(var = "SampleID")

# 3️⃣  Grab metadata & clean column names ---------------------------------------
meta_df <- colData(se) %>%
  as.data.frame() %>%
  rownames_to_column(var = "SampleID") %>%
  transmute(
    SampleID,
    age_group = age_category,
    bmi       = BMI,
    disease   = disease
  )

# 4️⃣  Join, create has_CVD label ----------------------------------------------
full_df <- abund_df %>%
  left_join(meta_df, by = "SampleID") %>%
  mutate(
    has_CVD = if_else(
      str_detect(tolower(disease), "atherosclerosis|cardiovascular|acvd"),
      1L, 0L
    )
  )

# 5️⃣  Write out ----------------------------------------------------------------
write.csv(full_df, "jie_2017_cvd_microbiome_full.csv", row.names = FALSE)

cat("✅ Saved: jie_2017_cvd_microbiome_full.csv\n")
