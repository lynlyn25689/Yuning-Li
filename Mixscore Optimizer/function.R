# =============================================================================
# functions.R - 墨西哥信贷模型评估自定义函数库
# =============================================================================

# ----------------------------- 读取 Parquet（带重试）----------------------------
#' 读取 Parquet 文件，支持重试机制
#' @param path 文件路径
#' @param columns 要选择的列，NULL 表示全部
#' @param loop_round 最大重试次数
#' @param wait_time 每次重试等待分钟数
read.parquet1 <- function(path, columns = NULL, loop_round = 6, wait_time = 5) {
  loop_cnt <- 0
  while (loop_cnt <= loop_round) {
    if (is.null(columns)) {
      data_my <- try(read_parquet(file = path), silent = TRUE)
    } else {
      data_my <- try(read_parquet(file = path, col_select = all_of(columns)), silent = TRUE)
    }
    if (inherits(data_my, "try-error")) {
      message(paste0('后台数据保存中，', wait_time, '分钟后重试...'))
      Sys.sleep(wait_time * 60)
      loop_cnt <- loop_cnt + 1
    } else {
      if (loop_cnt > 0) print(paste0('数据在第 ', loop_cnt, ' 次重试后成功读取'))
      break
    }
  }
  if (inherits(data_my, "try-error")) {
    stop(paste0('错误：已重试 ', loop_round, ' 次，仍未成功读取文件'))
  }
  data_my <- collapse::qDF(data_my)   # 转为标准 data.frame，若没有 collapse 包可改为 as.data.frame
  return(data_my)
}

# ----------------------------- KS 曲线数据生成 -----------------------------
#' 生成 KS 曲线的累积分布数据
ksLorenzStd <- function(predictions, labels) {
  labelOrdered <- labels[order(predictions, decreasing = TRUE)]
  popuPct <- (1:length(labelOrdered)) / length(labelOrdered)
  posCumPct <- cumsum(labelOrdered) / sum(labelOrdered)
  negCumPct <- cumsum(1 - labelOrdered) / sum(1 - labelOrdered)
  df <- data.frame(posCumPct, negCumPct, popuPct)
  return(df)
}

# ----------------------------- KS 曲线绘图 -----------------------------
#' 绘制 KS 曲线并标注 KS 值
ks_draw <- function(predictions, labels, models = "", addv = "") {
  df <- ksLorenzStd(predictions, labels)
  df1 <- df[, c('popuPct', 'posCumPct')]
  names(df1) <- c('Population', 'cumPct')
  df1$class <- 'pos'
  df2 <- df[, c('popuPct', 'negCumPct')]
  names(df2) <- c('Population', 'cumPct')
  df2$class <- 'neg'
  
  maxIntevPoint <- which.max(df1$cumPct - df2$cumPct)
  x1 <- df1$Population[maxIntevPoint]
  y1 <- df1$cumPct[maxIntevPoint]
  y2 <- df2$cumPct[maxIntevPoint]
  labelStr <- paste0("KS:", round((y1 - y2) * 100, 2))
  
  df_plot <- rbind(df1, df2)
  df_plot$class <- factor(df_plot$class, levels = c('pos', 'neg'))
  
  p <- ggplot(data = df_plot, aes(x = Population, y = cumPct, colour = class)) +
    geom_line() +
    labs(x = '%Population', y = '%Target', title = paste('KS Curve', models, addv)) +
    geom_abline(intercept = 0, slope = 1) +
    scale_x_continuous(limits = c(0, 1)) +
    scale_y_continuous(limits = c(0, 1)) +
    annotate('text', label = labelStr, x = 0.75, y = 0.15) +
    annotate('segment', x = x1, xend = x1, y = y1, yend = y2, colour = 'blue')
  return(p)
}

# ----------------------------- AUC 曲线绘图（备用） -----------------------------
#' 绘制 ROC 曲线并标注 AUC（需 pROC 包）
auc_draw <- function(predictions, labels) {
  if (!requireNamespace("pROC", quietly = TRUE)) {
    stop("请安装 pROC 包: install.packages('pROC')")
  }
  modelroc <- pROC::roc(labels, predictions, plot = TRUE, print.thres = TRUE, print.auc = TRUE)
  p <- plot(modelroc, print.auc = TRUE, auc.polygon = TRUE,
            grid = c(0.1, 0.2), grid.col = c("green", "red"),
            max.auc.polygon = TRUE, auc.polygon.col = "skyblue",
            print.thres = TRUE)
  return(p)
}

# =============================================================================
# 核心评估函数：合并 ks_auc_compute 与 ksauc_computer
# =============================================================================
#' 计算 KS 和 AUC，支持抽样和绘图
#' @param data 数据框
#' @param score_col 预测分数列名
#' @param target_col 目标变量列名（二分类，1 表示坏样本，0 表示好样本）
#' @param sample_frac 抽样比例（0~1），默认 1（全量）
#' @param draw 是否绘制 KS 曲线，'no' 不绘制，'yes' 绘制
#' @param addv 绘图的附加标题信息
#' @param seed 随机种子（保证抽样可复现），默认 NULL
#' @return 返回列表，包含 ks, auc, 以及用于绘图的 ggplot 对象（若 draw='yes'）
calc_ks_auc <- function(data, score_col, target_col,
                        sample_frac = 1, draw = 'no', addv = '',
                        seed = NULL) {
  # 筛选有效样本（分数和目标非负）
  data_use <- data[data[[target_col]] >= 0 & data[[score_col]] >= 0, ]
  
  # 抽样（若 sample_frac < 1）
  if (sample_frac < 1) {
    if (!is.null(seed)) set.seed(seed)
    data_use <- data_use[sample(nrow(data_use), size = floor(sample_frac * nrow(data_use))), ]
  }
  
  # 检查好坏样本数量
  bad_idx <- which(data_use[[target_col]] == 0)
  if (length(bad_idx) < 2) {
    warning("坏样本不足 2 个，无法计算 KS/AUC")
    return(list(ks = NA, auc = NA, plot = NULL))
  }
  
  pred <- prediction(data_use[[score_col]], data_use[[target_col]])
  perf <- performance(pred, "tpr", "fpr")
  ks <- round(max(attr(perf, 'y.values')[[1]] - attr(perf, 'x.values')[[1]]), 4)
  
  perf2 <- performance(pred, 'auc')
  auc <- round(as.numeric(perf2@y.values), 4)
  
  plot_obj <- NULL
  if (draw == 'yes') {
    plot_obj <- ks_draw(data_use[[score_col]], data_use[[target_col]], 
                        models = score_col, addv = addv)
    print(plot_obj)
  }
  
  return(list(ks = ks, auc = auc, plot = plot_obj))
}

# =============================================================================
# 逻辑回归变量相对重要性
# =============================================================================
#' 计算逻辑回归模型中各变量的相对重要性（基于相关系数矩阵）
relweights <- function(model_name, ...) {
  R <- cor(model_name$model)   # 使用模型框架中的数据计算相关系数
  nvar <- ncol(R)
  rxx <- R[2:nvar, 2:nvar]
  rxy <- R[2:nvar, 1]
  svd <- eigen(rxx)
  evec <- svd$vectors
  ev <- svd$values
  delta <- diag(sqrt(ev))
  lambda <- evec %*% delta %*% t(evec)
  lambdasq <- lambda^2
  beta <- solve(lambda) %*% rxy
  rsquare <- colSums(beta^2)
  rawwgt <- lambdasq %*% beta^2
  import <- round((rawwgt / rsquare), 2)
  lbls <- names(model_name$model[2:nvar])
  rownames(import) <- lbls
  colnames(import) <- 'weights'
  return(import)
}

# =============================================================================
# 枚举权重组合并计算 KS/AUC 及 Lift 变异系数
# =============================================================================
#' 生成所有权重组合（步长 step），计算各组合对三个目标的 KS/AUC，
#' 以及针对 target1 的 Lift CV 和单调性比例。
combine_models_ks <- function(data, model_vars, target1, target2, target3,
                              step = 0.1, preprocess = NULL,
                              na_action = c("omit", "zero", "fail"),
                              round_digits = 5, show_progress = TRUE,
                              n_bins = 20, sample_frac = 1) {
  na_action <- match.arg(na_action)
  
  # 检查变量存在性
  missing_vars <- setdiff(model_vars, names(data))
  if (length(missing_vars) > 0) {
    stop("以下变量在数据中不存在：", paste(missing_vars, collapse = ", "))
  }
  for (target in c(target1, target2, target3)) {
    if (!target %in% names(data)) stop("目标变量 ", target, " 不存在")
  }
  
  # 验证步长
  total_weight <- 1
  if (abs(step * round(total_weight / step) - total_weight) > 1e-8) {
    stop("步长 step 必须能整除 1（例如 0.1, 0.2）")
  }
  k <- as.integer(total_weight / step)
  
  n <- length(model_vars)
  
  # 预处理
  data_use <- data
  if (!is.null(preprocess)) {
    for (v in model_vars) {
      data_use[[v]] <- preprocess(data_use[[v]])
    }
  }
  
  # 缺失值处理
  if (na_action == "omit") {
    complete_idx <- complete.cases(data_use[, c(model_vars, target1, target2, target3)])
    data_use <- data_use[complete_idx, ]
    if (nrow(data_use) == 0) stop("删除缺失值后无有效数据")
  } else if (na_action == "zero") {
    for (v in model_vars) {
      data_use[[v]][is.na(data_use[[v]])] <- 0
    }
  } else if (na_action == "fail") {
    if (any(is.na(data_use[, model_vars]))) {
      stop("模型分数列存在缺失值，请设置 na_action = 'omit' 或 'zero'")
    }
  }
  
  # 生成所有非负整数组合（隔板法）
  total_positions <- k + n - 1
  all_comb <- utils::combn(total_positions, n - 1)
  n_combn <- ncol(all_comb)
  
  if (n_combn > 1e5) {
    warning("组合数超过 10 万 (", n_combn, ")，计算可能较慢")
  }
  
  # 预分配权重矩阵
  weight_mat <- matrix(0, nrow = n_combn, ncol = n)
  for (i in 1:n_combn) {
    pos <- all_comb[, i]
    x <- integer(n)
    x[1] <- pos[1] - 1
    if (n > 2) {
      for (j in 2:(n - 1)) {
        x[j] <- pos[j] - pos[j - 1] - 1
      }
    }
    x[n] <- total_positions - pos[n - 1]
    weight_mat[i, ] <- x * step
  }
  
  # 初始化进度条
  if (show_progress) {
    pb <- txtProgressBar(min = 0, max = n_combn, style = 3)
  }
  
  results_list <- vector("list", n_combn)
  
  for (i in 1:n_combn) {
    if (show_progress) setTxtProgressBar(pb, i)
    
    weights <- weight_mat[i, ]
    combined <- rep(0, nrow(data_use))
    for (j in 1:n) {
      combined <- combined + weights[j] * data_use[[model_vars[j]]]
    }
    combined <- round(combined, round_digits)
    temp_data <- data_use
    temp_data$combined_score <- combined
    
    # 调用统一函数 calc_ks_auc
    res1 <- calc_ks_auc(temp_data, "combined_score", target1, sample_frac = sample_frac)
    res2 <- calc_ks_auc(temp_data, "combined_score", target2, sample_frac = sample_frac)
    res3 <- calc_ks_auc(temp_data, "combined_score", target3, sample_frac = sample_frac)
    
    # 计算 Lift CV 和单调性（针对 target1）
    valid_idx <- which(temp_data[[target1]] >= 0)
    if (length(valid_idx) < n_bins) {
      lift_cv <- NA
      mono_ratio_fpd <- NA
    } else {
      score_valid <- temp_data$combined_score[valid_idx]
      target_valid <- temp_data[[target1]][valid_idx]
      
      breaks <- quantile(score_valid, probs = seq(0, 1, length.out = n_bins + 1), na.rm = TRUE)
      if (any(duplicated(breaks))) {
        bins <- as.numeric(cut(score_valid, breaks = n_bins, labels = FALSE))
      } else {
        bins <- as.numeric(cut(score_valid, breaks = breaks, include.lowest = TRUE))
      }
      
      bin_bad_rate <- tapply(target_valid == 0, bins, mean, na.rm = TRUE)
      overall_bad_rate <- mean(target_valid == 0, na.rm = TRUE)
      
      if (overall_bad_rate == 0 || any(is.na(bin_bad_rate))) {
        lift_cv <- NA
        mono_ratio_fpd <- NA
      } else {
        lift <- bin_bad_rate / overall_bad_rate
        lift_cv <- sd(lift, na.rm = TRUE) / mean(lift, na.rm = TRUE)
        
        bin_mean_score <- tapply(score_valid, bins, mean, na.rm = TRUE)
        sorted_bad_rate <- bin_bad_rate[order(bin_mean_score)]
        if (length(sorted_bad_rate) >= 2) {
          decrease_pairs <- sum(diff(sorted_bad_rate) < 0)
          mono_ratio_fpd <- decrease_pairs / (length(sorted_bad_rate) - 1)
        } else {
          mono_ratio_fpd <- NA
        }
      }
    }
    
    results_list[[i]] <- data.frame(
      t(weights),
      ks_fpd = res1$ks,
      auc_fpd = res1$auc,
      ks_cpd = res2$ks,
      auc_cpd = res2$auc,
      ks_spd = res3$ks,
      auc_spd = res3$auc,
      ks_avg = 0.5 * res1$ks + 0.5 * res2$ks,
      auc_avg = 0.5 * res1$auc + 0.5 * res2$auc,
      lift_cv_fpd = lift_cv,
      mono_ratio_fpd = mono_ratio_fpd
    )
  }
  
  if (show_progress) close(pb)
  
  results_df <- do.call(rbind, results_list)
  names(results_df)[1:n] <- model_vars
  return(results_df)
}

# =============================================================================
# 模型分数排序性分析（分箱统计）
# =============================================================================
#' 计算模型分数的等频分箱统计，用于评估排序性/单调性
#' @param test_data 数据框
#' @param p 分数列名
#' @param targetVar 目标变量列名（0=坏，1=好）
#' @param point 保留小数位数
#' @param sort 分箱数量（默认20）
#' @param print_sort 是否打印结果
#' @param print_chart 是否绘制条形图
#' @param addv 绘图附加标题
#' @return 数据框，包含各分箱的统计指标
model_result_sort <- function(test_data, p, targetVar, point = 2, sort = 20,
                              print_sort = 'yes', print_chart = 'yes',
                              addv = '') {
  # 筛选有效样本
  test_data <- test_data[test_data[, targetVar] >= 0 & test_data[, p] >= 0, ]
  if (nrow(test_data) == 0) {
    warning("无有效样本")
    return(NULL)
  }
  
  avg_target <- c()
  p_fenwei <- c()
  num_target <- c()
  accumulate_target <- c()
  p_fenwei_min <- c()
  avg_acc_target <- c()
  accumulate_rt <- c()
  accumulate_lift <- c()
  
  aaa <- test_data[, c(p, targetVar)]
  aaa <- aaa[order(aaa[, p]), ]
  count_data <- nrow(aaa)
  grade_no <- floor(count_data / sort)
  sum_bad <- sum(test_data[, targetVar] == 0)
  mean_good <- mean(test_data[, targetVar])  # 注意这里可能是目标均值（好样本比例）
  
  for (i in 1:sort) {
    begin <- grade_no * (i - 1) + 1
    if (i < sort) {
      end <- grade_no * i
    } else {
      end <- nrow(aaa)
    }
    b <- round(mean(aaa[begin:end, targetVar]), point)
    ttt <- round(sum(1 - aaa[begin:end, targetVar]), point)
    p1 <- round(aaa[end, p], point)
    p1_min <- round(aaa[begin, p], point)
    
    p_fenwei <- c(p_fenwei, p1)
    p_fenwei_min <- c(p_fenwei_min, p1_min)
    avg_target <- c(avg_target, b)
    num_target <- c(num_target, ttt)
    sum_num_target <- sum(num_target)
    accumulate_target <- c(accumulate_target, sum_num_target)
    
    accumulate_count <- grade_no * i
    if (i == sort) accumulate_count <- nrow(aaa)
    accumulate_rt_current <- round(sum_num_target / accumulate_count, point)
    accumulate_lift_current <- round(accumulate_rt_current / (1 - mean_good), point)
    accumulate_rt <- c(accumulate_rt, accumulate_rt_current)
    accumulate_lift <- c(accumulate_lift, accumulate_lift_current)
    
    begin_acc <- grade_no * i + 1
    end_acc <- nrow(aaa)
    c <- round(mean(aaa[begin_acc:end_acc, targetVar]), point)
    avg_acc_target <- c(avg_acc_target, c)
  }
  
  result_sort <- data.frame(
    rbind(
      p_fenwei,
      p_fenwei_min,
      num_target,
      lift = round((1 - avg_target) / (1 - mean_good), point),
      target_rt = (1 - avg_target),
      accumulate_target,
      accumulate_bad_rt = round(accumulate_target / sum_bad, 2),
      acc_bl = round((1 - avg_acc_target) / (1 - mean_good), point),
      accumulate_rt = accumulate_rt,
      accumulate_lift = accumulate_lift
    )
  )
  names(result_sort) <- paste0((1:sort) * (round(100 / sort, 1)), '%')
  
  # 绘制条形图（可选）
  if (print_chart == 'yes') {
    chart <- plot(1:length(names(result_sort)), result_sort[4,],
                  main = paste(p, addv), xlab = "Bin", ylab = "Lift")
    labels <- as.character(result_sort[4, ])
    text(1:length(names(result_sort)), result_sort[4, ], labels,
         pos = 4, offset = 0.5, font = 5, cex = 0.7)
    print(chart)
  }
  
  if (print_sort == 'yes') {
    print(result_sort)
    return(result_sort)
  }
  return(result_sort)
}

# =============================================================================
# 辅助函数：计算十分位风险（用于额度加权分析）
# =============================================================================
#' 根据模型阈值筛选通过样本，按分数十分位统计逾期率和金额
calc_decile_risk <- function(df, model_col, threshold, target_cols, avg_amt_df) {
  passed_df <- df[df[[model_col]] >= threshold, ]
  if (nrow(passed_df) == 0) return(NULL)
  
  passed_df <- passed_df[order(passed_df[[model_col]]), ]
  passed_df$decile <- dplyr::ntile(passed_df[[model_col]], 10)
  passed_df <- passed_df %>%
    mutate(decile = sprintf("Q%02d", decile))
  
  # 匹配平均额度（按 decile 匹配 credit_score_range）
  passed_df <- passed_df %>%
    left_join(avg_amt_df, by = c("decile" = "credit_score_range"))
  passed_df$avg_amt_cl <- ifelse(is.na(passed_df$avg_amt_cl),
                                 passed_df$amt_cl,
                                 passed_df$avg_amt_cl)
  
  decile_res <- passed_df %>%
    group_by(decile) %>%
    summarise(
      n = n(),
      total_loan = sum(avg_amt_cl, na.rm = TRUE),
      across(all_of(target_cols),
             ~ sum(. == 0, na.rm = TRUE) / sum(. >= 0, na.rm = TRUE),
             .names = "{.col}")
    ) %>%
    ungroup() %>%
    mutate(across(all_of(target_cols),
                  ~ total_loan * .,
                  .names = "bad_amt_{.col}"))
  
  total_res <- decile_res %>%
    summarise(
      n = sum(n),
      total_loan = sum(total_loan),
      across(starts_with("bad_amt_"), sum)
    ) %>%
    mutate(decile = "Total")
  
  for (t in target_cols) {
    total_res[[t]] <- sum(passed_df[[t]] == 0, na.rm = TRUE) /
      sum(passed_df[[t]] >= 0, na.rm = TRUE)
  }
  
  bind_rows(decile_res, total_res)
}
