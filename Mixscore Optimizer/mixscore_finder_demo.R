# =============================================================================
# analysis_demo.R - 墨西哥信贷模型评估完整示例
# 依赖：functions.R 中的自定义函数，以及以下 R 包：
#   ROCR, dplyr, tidyr, ggplot2, stringr, openxlsx, arrow, collapse, lubridate
# =============================================================================

# 1. 加载所有依赖包 ------------------------------------------------
library(ROCR)
library(dplyr)
library(tidyr)
library(ggplot2)
library(stringr)
library(openxlsx)
library(arrow)
library(collapse)
library(lubridate)

# 2. 加载自定义函数 ------------------------------------------------
source("functions.R")  

# 3. 数据准备 -------------------------------------------------------
mexico_credit <- 
mexico_cash   <- 

# 现金数据清洗（仅首次借款）
mexico_cash <- mexico_cash %>%
  filter(is_first_cash == 1, trade_amt > 0, day >= '20260501') %>%
  select(no_appl, no_appl_tixian, id_unqp, time_inst_tixian = time_inst,
         target_fpd1_tixian = target_fpd1,
         target_cpd1_tixian = target_cpd1,
         target_spd1_tixian = target_spd1,
         target_tpd1_tixian = target_tpd1,
         trade_amt,
         spl_prin_sum_cash = spl_prin_sum) %>%
  group_by(no_appl) %>%
  slice_min(time_inst_tixian, n = 1, with_ties = FALSE) %>%
  ungroup()

# 授信数据筛选并合并
oot_start_date <- '20260601'

mexico_credit <- mexico_credit %>%
  left_join(mexico_cash, by = 'no_appl') %>%
  mutate(mixscore_use = mixscore_use)

# 4. 分数分布（十分位数）------------------------------------------
mexico_credit_use <- mexico_credit %>% filter(rnd_fnlres == 'CP', day >= '20260701')
var_list <- c("dpmaf429fnscore", "dpmaf425fnscore", "dpmaf430fnscore", "dpmaf427fnscore",
              "dpmaf487fnscore", "dpmaf488fnscore", "dpmaf489fnscore", "dpmaf490fnscore",
              "dpmaf491fnscore", "dpmaf492fnscore", "dpmaf493fnscore", "dpmaf494fnscore")
probs <- seq(0.1, 0.9, 0.1)
quantile_list <- lapply(var_list, function(v) {
  quantile(mexico_credit_use[[v]], probs = probs, na.rm = TRUE)
})
df_thresholds <- do.call(rbind, quantile_list)
rownames(df_thresholds) <- var_list
colnames(df_thresholds) <- paste0("q", round(probs * 100))
print("分数十分位数：")
print(df_thresholds)

# 5. 相关性分析 ----------------------------------------------------
model_vars_corr <- c("dpmaf487fnscore", "dpmaf488fnscore", "dpmaf489fnscore",
                     "dpmaf490fnscore", "dpmaf491fnscore", "dpmaf492fnscore",
                     "dpmaf493fnscore", "dpmaf494fnscore")
data_corr <- mexico_credit %>%
  filter(rnd_fnlres == 'CP', day >= 20260601,
         if_all(all_of(model_vars_corr), ~ . > 0)) %>%
  select(all_of(model_vars_corr))
cor_matrix <- cor(data_corr, use = "complete.obs")
print("相关矩阵：")
print(cor_matrix)

# 6. 枚举权重组合（演示用，仅取前3个变量）-------------------------
model_vars_enum <- c("dpmaf425fnscore", "dpmaf429fnscore", "dpmaf430fnscore")  
data_enum <- mexico_credit %>% 
  filter(target_fpd1 >= 0) %>%   # 根据原代码条件
  select(all_of(model_vars_enum), target_fpd1, target_cpd1, target_fpd10)

# 定义预处理函数（将小于0的值替换为0.4）
preproc <- function(x) ifelse(x < 0, 0.4, x)

# 运行枚举（步长0.1，注意组合数可能很大，此处仅演示，实际可减少变量数量或增大步长）
# 为演示，我们只取前3个变量以减少计算量（实际使用时请注释掉下行）
model_vars_enum_demo <- model_vars_enum[1:3]   # 仅演示用，实际可恢复为全部

# 调用函数（注意 sample_frac 设为1以便使用全量数据，原代码未抽样）
results_enum <- combine_models_ks(
  data        = data_enum,
  model_vars  = model_vars_enum_demo,   # 演示用，正式请用 model_vars_enum
  target1     = "target_fpd1",
  target2     = "target_cpd1",
  target3     = "target_fpd10",
  step        = 0.1,
  preprocess  = preproc,
  na_action   = "omit",
  sample_frac = 1,
  show_progress = TRUE
)

# 查看按 ks_fpd 降序排列的前几行
head(results_enum[order(-results_enum$ks_fpd), ])

# 保存结果到 Excel（可选）
wb <- createWorkbook()
addWorksheet(wb, "weight_combinations")
writeData(wb, sheet = "weight_combinations", results_enum, startCol = 1, startRow = 1)
saveWorkbook(wb, file = "model_performance_metrics.xlsx", overwrite = TRUE)

# 7. 示例：逻辑回归组合（穷举特征组合）--------------------------------------
# 定义使用的特征（与前面一致）
model_use_lr <- c("dpmaf425fnscore", "dpmaf429fnscore", "dpmaf430fnscore", 
                  "dpmaf487fnscore", "dpmaf489fnscore",
                  "dpmaf491fnscore", "dpmaf492fnscore", "dpmaf493fnscore", 
                  "dpmaf494fnscore")

# 准备数据（OOT 样本，目标变量之一，这里以 target_fpd1 为例）
data_lr <- mexico_credit %>% 
  filter(target_fpd1 >= 0, day >= 20260601) %>%
  select(all_of(model_use_lr), target_fpd1, target_cpd1, target_fpd10)

# 小于0填充0.4
for (var in model_use_lr) {
  data_lr[[var]] <- ifelse(data_lr[[var]] < 0, 0.4, data_lr[[var]])
}

# 目标变量列表
target_vars <- c('target_fpd1', 'target_cpd1', 'target_fpd10')

# 存储结果的空数据框
modelauc_sigle_res <- NULL

# 循环目标变量
for (k in target_vars) {
  cat("\n===== 开始处理目标变量:", k, "=====\n")
  data_use1 <- data_lr
  data_use1 <- data_use1[data_use1[[k]] >= 0, ]
  
  # 循环特征组合个数（原代码从4开始到变量总数，此处演示只取4个）
  # 正式可修改为 for(i in 4:length(model_use_lr))
  for (i in 4:4) {   # 演示仅 i=4，正式时取消注释下一行并注释此行
    # for (i in 4:length(model_use_lr)) {
    comb_matrix <- combn(model_use_lr, i)
    n_comb <- ncol(comb_matrix)
    cat("  特征个数 =", i, "，组合数 =", n_comb, "\n")
    
    for (j in 1:n_comb) {
      if (j %% 10 == 0 || j == n_comb) {
        cat("    正在处理组合", j, "/", n_comb, "\n")
      }
      
      # 拟合逻辑回归
      myglm <- glm(data_use1[[k]] ~ ., 
                   data = data_use1[, comb_matrix[, j], drop = FALSE], 
                   family = binomial(link = "logit"))
      
      # 计算相对重要性
      lm_coff <- round(sort(myglm$coefficients), 2)
      var_importance <- relweights(myglm)
      var_importance <- var_importance[order(-var_importance$weights), ]
      
      # 构造加权分数表达式（按重要性加权）
      text <- paste(var_importance$weights, rownames(var_importance), sep = '*', collapse = '+')
      
      # 计算加权分数
      data_use1$score_for <- eval(parse(text = text))
      
      # 调用 calc_ks_auc 计算性能
      res <- calc_ks_auc(data_use1, "score_for", target_col = k, sample_frac = 1)
      
      # 记录结果
      temp_res <- data.frame(
        target = k,
        features = paste(comb_matrix[, j], collapse = "+"),
        modelweight = text,
        orgmodelweight = paste(lm_coff, names(lm_coff), sep = '*', collapse = '+'),
        ks = res$ks,
        auc = res$auc
      )
      modelauc_sigle_res <- rbind(modelauc_sigle_res, temp_res)
    }
  }
  cat("===== 完成目标变量:", k, "=====\n\n")
}

# 查看结果
print(head(modelauc_sigle_res[order(-modelauc_sigle_res$auc), ]))

# 添加二元标记（原代码中的特征存在标记）
for (var in model_use_lr) {
  modelauc_sigle_res[[var]] <- ifelse(grepl(var, modelauc_sigle_res$modelweight), 1, 0)
}

# 按 AUC 降序排列
modelauc_sigle_res <- arrange(modelauc_sigle_res, -auc)

# 保存结果
wb2 <- createWorkbook()
addWorksheet(wb2, "modelauc_sigle_res")
writeData(wb2, sheet = "modelauc_sigle_res", modelauc_sigle_res, startCol = 1, startRow = 1, rowNames = FALSE)
saveWorkbook(wb2, file = paste0("逻辑回归modelauc_sigle_res", 
                                format(Sys.time() + 14 * 3600, "%Y%m%d_%H%M"), ".xlsx"), overwrite = TRUE)


# 8. 多模型效果比较（KS/AUC）--------------------------------------
# 定义要比较的模型列表（根据实际生成的新变量）
mixscore_list <- as.list(c("mixscore_use", paste0("mixscore_new_", sprintf("%02d", 1:85))))

# 准备不同时间/样本的数据集
oot_start_date_num <- 20260601
mexico_credit_first_random <- mexico_credit_first %>%
  filter(day >= oot_start_date_num, target_fpd1 >= 0, price_type == 'new_price_type')
mexico_credit_first_random_fpd10 <- mexico_credit_first %>%
  filter(day >= oot_start_date_num, target_fpd10 >= 0, price_type == 'new_price_type')
mexico_credit_first_random_cpd <- mexico_credit_first %>%
  filter(day >= 20260501, day < 20260601, target_tpd1_tixian >= 0, price_type == 'new_price_type')

ks_auc_res <- NULL
for (pp in mixscore_list) {
  count_nonneg <- sum(mexico_credit_first_random[[pp]] >= 0, na.rm = TRUE)
  if (count_nonneg < 10) next
  
  # 使用 calc_ks_auc 替代旧函数
  res_fpd <- calc_ks_auc(mexico_credit_first_random, score_col = pp, target_col = 'target_fpd1')
  res_fpd10 <- calc_ks_auc(mexico_credit_first_random_fpd10, score_col = pp, target_col = 'target_fpd10')
  res_cpd_full <- calc_ks_auc(mexico_credit_first_random, score_col = pp, target_col = 'target_cpd1')
  res_cpd_sub <- calc_ks_auc(mexico_credit_first_random_cpd, score_col = pp, target_col = 'target_cpd1')
  res_tpd <- calc_ks_auc(mexico_credit_first_random_cpd, score_col = pp, target_col = 'target_tpd1_tixian')
  res_term3 <- calc_ks_auc(mexico_credit_first_random_cpd, score_col = pp, target_col = 'target_term3cpd1')
  
  row_values <- c(pp,
                  res_fpd$ks, res_fpd$auc,
                  res_fpd10$ks, res_fpd10$auc,
                  res_cpd_full$ks, res_cpd_full$auc,
                  res_cpd_sub$ks, res_cpd_sub$auc,
                  res_tpd$ks, res_tpd$auc,
                  res_term3$ks, res_term3$auc,
                  sum(mexico_credit_first_random[[pp]] > 0, na.rm = TRUE),
                  sum(mexico_credit_first_random_fpd10[[pp]] > 0, na.rm = TRUE),
                  sum(mexico_credit_first_random_cpd[[pp]] > 0, na.rm = TRUE))
  ks_auc_res <- rbind(ks_auc_res, row_values)
}
colnames(ks_auc_res) <- c("model_name", "ks_fpd", "auc_fpd", "ks_fpd10", "auc_fpd10",
                          "ks_cpd1_full", "auc_cpd1_full", "ks_cpd_sub", "auc_cpd_sub",
                          "ks_tpd1_tixian", "auc_tpd1_tixian", "ks_term3cpd1", "auc_term3cpd1",
                          "count_pp", "count_pp_fpd10", "count_pp_cpd")
ks_auc_res_df <- as.data.frame(ks_auc_res, stringsAsFactors = FALSE)
ks_auc_res_df[, -1] <- lapply(ks_auc_res_df[, -1], as.numeric)
ks_auc_res_df <- ks_auc_res_df[order(ks_auc_res_df$model_name), ]
View(ks_auc_res_df)

# 9. 排序性分析（model_result_sort）--------------------------------
sort_num <- 20
sort_res_fpd <- NULL
for (pp in mixscore_list) {
  res <- model_result_sort(mexico_credit_first_random, pp, 'target_fpd1',
                           point = 3, print_chart = 'yes', sort = sort_num)
  if (!is.null(res)) {
    res$coefname <- pp
    sort_res_fpd <- rbind(sort_res_fpd, res[, c('coefname', names(res)[1:sort_num])])
  }
}
# 同样可做 cpd 排序性，此处省略重复代码

# 10. 同拒绝分位风险（额度加权）-----------------------------------
# 先计算各信用分段的平均额度
# 计算平均放款额度（按 credit_score_range）
avg_amt_by_range <- mexico_credit %>%
  filter(day >= 20260710, credit_score_range != -9999, res_audit == 'accept') %>%
  group_by(credit_score_range) %>%
  summarise(
    avg_amt_cl = mean(amt_cl, na.rm = TRUE),
    count = n()
  ) %>%
  arrange(credit_score_range)

# 计算 FPD 风险
fpd_targets <- c("target_fpd1", "target_cpd1", "target_fpd10")
# 首先计算各模型在 FPD 数据集上的阈值（使用 mixscore_use 的 69% 分位）
p_fpd <- ecdf(mexico_credit_use_fpd$mixscore_use)(0.69)
result_df_fpd <- data.frame(
  model = model_vars,
  value = sapply(model_vars, function(col) {
    round(quantile(mexico_credit_use_fpd[[col]], probs = p_fpd, na.rm = TRUE), 2)
  })
)
fpd_list <- list()
for (model in model_vars) {
  thresh <- result_df_fpd$value[result_df_fpd$model == model]
  if (length(thresh) == 0 || is.na(thresh)) next
  res <- calc_decile_risk(mexico_credit_use_fpd, model, thresh, 
                          fpd_targets, avg_amt_by_range)
  if (!is.null(res)) {
    res$model <- model
    fpd_list[[model]] <- res
  }
}
fpd_risk_by_decile <- bind_rows(fpd_list) %>%
  select(model, decile, n, total_loan, 
         all_of(fpd_targets), 
         starts_with("bad_amt_"))

# 计算 CPD 风险
cpd_targets <- c("target_cpd1", "target_spd1_tixian", "target_tpd1_tixian", "target_term3cpd1")
p_cpd <- ecdf(mexico_credit_use_cpd$mixscore_use)(0.69)
result_df_cpd <- data.frame(
  model = model_vars,
  value = sapply(model_vars, function(col) {
    round(quantile(mexico_credit_use_cpd[[col]], probs = p_cpd, na.rm = TRUE), 2)
  })
)
cpd_list <- list()
for (model in model_vars) {
  thresh <- result_df_cpd$value[result_df_cpd$model == model]
  if (length(thresh) == 0 || is.na(thresh)) next
  res <- calc_decile_risk(mexico_credit_use_cpd, model, thresh, 
                          cpd_targets, avg_amt_by_range)
  if (!is.null(res)) {
    res$model <- model
    cpd_list[[model]] <- res
  }
}
cpd_risk_by_decile <- bind_rows(cpd_list) %>%
  select(model, decile, n, total_loan, 
         all_of(cpd_targets), 
         starts_with("bad_amt_"))

# 合并风险结果
combined_risk <- left_join(
  fpd_risk_by_decile,
  cpd_risk_by_decile,
  by = c("model", "decile"),
  suffix = c("_fpd", "_cpd")
)
View(combined_risk)

# 保存到 Excel
wb_risk <- createWorkbook()
addWorksheet(wb_risk, "combined_risk")
writeData(wb_risk, sheet = "combined_risk", combined_risk, startCol = 1, startRow = 1, rowNames = FALSE)
openxlsx::saveWorkbook(wb_risk, paste0("~/mix效果比较", format(Sys.time() + 14 * 3600, "%Y%m%d_%H%M"), ".xlsx"), overwrite = TRUE)
