# STAT 5820 Project
# Structural Change Detection Using Supervised Learning Residuals

library("ggplot2")
library(randomForest)
library(tidyr)
library(dplyr)

rm(list = ls())

set.seed(123)

dir.create("figures", showWarnings = FALSE)
dir.create("slides", showWarnings = FALSE)

# -----------------------------
# 1. Load real dataset D
# -----------------------------

# Built-in real dataset in R
data(airquality)

# Look at the dataset
head(airquality)
summary(airquality)

# Remove rows with missing values
D <- na.omit(airquality)

# Check cleaned dataset
head(D)
dim(D)


# -----------------------------
# 2. Train regression function f on real dataset D
# -----------------------------

# Fit linear regression model using the real dataset
fit_lm <- lm(Ozone ~ Solar.R + Wind + Temp + Month + Day, data = D)

# Look at model output
summary(fit_lm)

# Get fitted values from the learned regression function f
f_hat <- predict(fit_lm, newdata = D)

# Create a data frame for plotting
plot_df <- data.frame(
  Actual = D$Ozone,
  Fitted = f_hat
)

# Compare actual Ozone values with fitted values
p1 <- ggplot(plot_df, aes(x = Actual, y = Fitted)) +
  geom_point(shape = 1, size = 3, color = "steelblue4", stroke = 1) +
  geom_abline(slope = 1, intercept = 0,
              color = "red", linetype = "dashed", linewidth = 1) +
  labs(
    title = "Actual vs Fitted Ozone Values",
    subtitle = "Linear regression fit based on the airquality dataset",
    x = "Actual Ozone",
    y = "Fitted Ozone"
  ) +
  theme_classic(base_size = 14) +
  theme(
    plot.title = element_text(face = "bold"),
    plot.subtitle = element_text(size = 12)
  )

print(p1)

ggsave("figures/actual_vs_fitted_ozone.png",
       plot = p1, width = 8, height = 6, dpi = 300)

# -----------------------------
# 3. Generate change-point data using learned f
# -----------------------------

set.seed(123)

# Number of observations in generated dataset
n_new <- 2000

# True change point
change_point <- 1000

# Time index
time <- 1:n_new

# Predictor variables from the real dataset D
# We sample rows from D with replacement to create new X values
X_new <- D[sample(1:nrow(D), size = n_new, replace = TRUE),
           c("Solar.R", "Wind", "Temp", "Month", "Day")]

# Use the learned regression function f to get baseline predictions
f_values <- predict(fit_lm, newdata = X_new)

# Create random noise
noise <- rnorm(n_new, mean = 0, sd = 10)

# Standardize Temp so the added x^2 change is not too large
Temp_std <- (X_new$Temp - mean(X_new$Temp)) / sd(X_new$Temp)

# Generate response Y
Y_new <- rep(NA, n_new)

# Unchanged part: Y = f(X) + noise
Y_new[time <= change_point] <- f_values[time <= change_point] +
  noise[time <= change_point]

# Changed part: Y = f(X) + 15 * Temp_std^2 + noise
Y_new[time > change_point] <- f_values[time > change_point] +
  15 * Temp_std[time > change_point]^2 +
  noise[time > change_point]

# Put generated data into one data frame
cp_data <- data.frame(
  time = time,
  Y = Y_new,
  X_new
)

# change row names from airquality to cp_data
rownames(cp_data) <- NULL

# Check the generated data
head(cp_data)
dim(cp_data)


# -----------------------------
# 4. Plot generated change-point data
# -----------------------------

p2 <- ggplot(cp_data, aes(x = time, y = Y)) +
  geom_line(color = "steelblue4", linewidth = 0.6) +
  geom_vline(xintercept = change_point,
             color = "red", linetype = "dashed", linewidth = 1) +
  labs(
    title = "Generated Data with Known Change Point",
    subtitle = "First part generated from f(X) + noise; second part generated from modified function g(X) + noise",
    x = "Time",
    y = "Generated Response Y"
  ) +
  theme_classic(base_size = 14) +
  theme(
    plot.title = element_text(face = "bold"),
    plot.subtitle = element_text(size = 11)
  )

print(p2)

ggsave("figures/generated_change_point_data.png",
       plot = p2, width = 9, height = 5, dpi = 300)


# -----------------------------
# 5. Fit model on unchanged part and compute residuals
# -----------------------------

# Training data: first 1000 observations (unchanged part)
train_data <- cp_data[1:change_point, ]

# Fit linear regression model on unchanged part
fit_cp_lm <- lm(Y ~ Solar.R + Wind + Temp + Month + Day, data = train_data)

# Look at model summary
summary(fit_cp_lm)

# Predict on the full dataset
cp_data$pred_lm <- predict(fit_cp_lm, newdata = cp_data)

# Residuals
cp_data$resid_lm <- cp_data$Y - cp_data$pred_lm
cp_data$abs_resid_lm <- abs(cp_data$resid_lm)

# Quick check
head(cp_data[, c("time", "Y", "pred_lm", "resid_lm", "abs_resid_lm")])


# -----------------------------
# 6. Plot absolute residuals over time
# -----------------------------


p3 <- ggplot(cp_data, aes(x = time, y = abs_resid_lm)) +
  geom_point(color = "steelblue4", alpha = 0.5, size = 1.2) +
  geom_vline(xintercept = change_point,
             color = "red", linetype = "dashed", linewidth = 1) +
  labs(
    title = "Absolute Residuals from Linear Regression",
    subtitle = "Model trained on the unchanged portion of the data",
    x = "Time",
    y = "Absolute Residual"
  ) +
  theme_classic(base_size = 14) +
  theme(
    plot.title = element_text(face = "bold"),
    plot.subtitle = element_text(size = 11)
  )

print(p3)

ggsave("figures/absolute_residuals_lm.png",
       plot = p3, width = 9, height = 5, dpi = 300)


# -----------------------------
# 7. Plot smoothed residual trend
# -----------------------------

p4 <- ggplot(cp_data, aes(x = time, y = abs_resid_lm)) +
  geom_point(color = "steelblue4", alpha = 0.25, size = 1) +
  geom_smooth(se = FALSE, color = "darkorange", linewidth = 1.2) +
  geom_vline(xintercept = change_point,
             color = "red", linetype = "dashed", linewidth = 1) +
  labs(
    title = "Residual Trend Over Time",
    subtitle = "Smoothed absolute residuals highlight the change point",
    x = "Time",
    y = "Absolute Residual"
  ) +
  theme_classic(base_size = 14) +
  theme(
    plot.title = element_text(face = "bold"),
    plot.subtitle = element_text(size = 11)
  )

print(p4)

ggsave("figures/smoothed_residuals_lm.png",
       plot = p4, width = 9, height = 5, dpi = 300)


# -----------------------------
# 8. Compare residuals before and after the change point
# -----------------------------

# Create a group variable
cp_data$period <- ifelse(cp_data$time <= change_point, "Before", "After")

# Mean absolute residual in each period
tapply(cp_data$abs_resid_lm, cp_data$period, mean)

# Median absolute residual in each period
tapply(cp_data$abs_resid_lm, cp_data$period, median)

# Standard deviation of absolute residuals in each period
tapply(cp_data$abs_resid_lm, cp_data$period, sd)

# Place results in table
residual_summary <- data.frame(
  Period = c("Before", "After"),
  Mean = c(
    mean(cp_data$abs_resid_lm[cp_data$period == "Before"]),
    mean(cp_data$abs_resid_lm[cp_data$period == "After"])
  ),
  Median = c(
    median(cp_data$abs_resid_lm[cp_data$period == "Before"]),
    median(cp_data$abs_resid_lm[cp_data$period == "After"])
  ),
  SD = c(
    sd(cp_data$abs_resid_lm[cp_data$period == "Before"]),
    sd(cp_data$abs_resid_lm[cp_data$period == "After"])
  )
)

print(residual_summary)


# Boxplot of before and after cp
# Switch before and after
cp_data$period <- factor(cp_data$period, levels = c("Before", "After"))
p5 <- ggplot(cp_data, aes(x = period, y = abs_resid_lm, fill = period)) +
  geom_boxplot(alpha = 0.8) +
  labs(
    title = "Absolute Residuals Before and After the Change Point",
    subtitle = "Linear regression model trained on the unchanged portion",
    x = "",
    y = "Absolute Residual"
  ) +
  theme_classic(base_size = 14) +
  theme(
    plot.title = element_text(face = "bold"),
    plot.subtitle = element_text(size = 11),
    legend.position = "none"
  )

print(p5)

ggsave("figures/boxplot_residuals_before_after_lm.png",
       plot = p5, width = 7, height = 5, dpi = 300)

# Quick test if mean absolute residual differs before and after cp
t.test(abs_resid_lm ~ period, data = cp_data)


# -----------------------------
# 9. Random forest on unchanged part
# -----------------------------

fit_rf <- randomForest(
  Y ~ Solar.R + Wind + Temp + Month + Day,
  data = train_data,
  ntree = 300
)

print(fit_rf)

# Predict on full dataset
cp_data$pred_rf <- predict(fit_rf, newdata = cp_data)

# Residuals
cp_data$resid_rf <- cp_data$Y - cp_data$pred_rf
cp_data$abs_resid_rf <- abs(cp_data$resid_rf)

# Quick check
head(cp_data[, c("time", "Y", "pred_rf", "resid_rf", "abs_resid_rf")])


# Random forest smoothed residual plot
p7 <- ggplot(cp_data, aes(x = time, y = abs_resid_rf)) +
  geom_point(color = "steelblue4", alpha = 0.2, size = 1) +
  geom_smooth(se = FALSE, color = "darkorange", linewidth = 1.2) +
  geom_vline(xintercept = change_point,
             color = "red", linetype = "dashed", linewidth = 1) +
  labs(
    title = "Residual Trend Over Time",
    subtitle = "Random forest absolute residuals",
    x = "Time",
    y = "Absolute Residual"
  ) +
  theme_classic(base_size = 14) +
  theme(
    plot.title = element_text(face = "bold"),
    plot.subtitle = element_text(size = 11)
  )

print(p7)

ggsave("figures/smoothed_residuals_rf.png",
       plot = p7, width = 9, height = 5, dpi = 300)

# Numerical summary
rf_summary <- data.frame(
  Period = c("Before", "After"),
  Mean = c(
    mean(cp_data$abs_resid_rf[cp_data$period == "Before"]),
    mean(cp_data$abs_resid_rf[cp_data$period == "After"])
  ),
  Median = c(
    median(cp_data$abs_resid_rf[cp_data$period == "Before"]),
    median(cp_data$abs_resid_rf[cp_data$period == "After"])
  ),
  SD = c(
    sd(cp_data$abs_resid_rf[cp_data$period == "Before"]),
    sd(cp_data$abs_resid_rf[cp_data$period == "After"])
  )
)

print(rf_summary)

t.test(abs_resid_rf ~ period, data = cp_data)

# -----------------------------
# 10. Combined residual trend plot by model
# -----------------------------

# Long format for the two smooth curves
resid_long <- cp_data %>%
  select(time, abs_resid_lm, abs_resid_rf) %>%
  pivot_longer(
    cols = c(abs_resid_lm, abs_resid_rf),
    names_to = "Model",
    values_to = "Absolute_Residual"
  ) %>%
  mutate(
    Model = case_when(
      Model == "abs_resid_lm" ~ "Linear Regression",
      Model == "abs_resid_rf" ~ "Random Forest"
    )
  )

# One background point cloud + two smooth curves
p8 <- ggplot() +
  geom_point(
    data = cp_data,
    aes(x = time, y = abs_resid_lm),
    color = "steelblue4",
    alpha = 0.22,
    size = 1.2
  ) +
  geom_smooth(
    data = resid_long,
    aes(x = time, y = Absolute_Residual, color = Model),
    se = FALSE,
    linewidth = 1.7
  ) +
  geom_vline(
    xintercept = change_point,
    color = "red",
    linetype = "dashed",
    linewidth = 1
  ) +
  labs(
    title = "Residual Trends by Model",
    subtitle = "Both models show larger residuals after the known change point at t = 1000",
    x = "Time",
    y = "Absolute Residual",
    color = "Model"
  ) +
  theme_classic(base_size = 14) +
  theme(
    plot.title = element_text(face = "bold", size = 18),
    plot.subtitle = element_text(size = 12),
    legend.position = c(0.03, 0.97),
    legend.justification = c(0, 1),
    legend.background = element_rect(fill = "white", color = "black"),
    legend.key = element_rect(fill = "white", color = NA)
  )

print(p8)

ggsave(
  "figures/residual_trends_by_model_singleplot.png",
  plot = p8,
  width = 9.5,
  height = 5.5,
  dpi = 300
)


# -----------------------------
# 10. Combined comparison table
# -----------------------------

comparison_table <- data.frame(
  Model = c("Linear Regression", "Linear Regression",
            "Random Forest", "Random Forest"),
  Period = c("Before", "After", "Before", "After"),
  Mean = c(
    mean(cp_data$abs_resid_lm[cp_data$period == "Before"]),
    mean(cp_data$abs_resid_lm[cp_data$period == "After"]),
    mean(cp_data$abs_resid_rf[cp_data$period == "Before"]),
    mean(cp_data$abs_resid_rf[cp_data$period == "After"])
  ),
  Median = c(
    median(cp_data$abs_resid_lm[cp_data$period == "Before"]),
    median(cp_data$abs_resid_lm[cp_data$period == "After"]),
    median(cp_data$abs_resid_rf[cp_data$period == "Before"]),
    median(cp_data$abs_resid_rf[cp_data$period == "After"])
  ),
  SD = c(
    sd(cp_data$abs_resid_lm[cp_data$period == "Before"]),
    sd(cp_data$abs_resid_lm[cp_data$period == "After"]),
    sd(cp_data$abs_resid_rf[cp_data$period == "Before"]),
    sd(cp_data$abs_resid_rf[cp_data$period == "After"])
  )
)

print(comparison_table)


# -----------------------------
# 11. Comparison plot
# -----------------------------

# Make sure Period order is Before then After
comparison_table$Period <- factor(comparison_table$Period,
                                  levels = c("Before", "After"))

p9 <- ggplot(comparison_table, aes(x = Model, y = Mean, fill = Period)) +
  geom_col(position = "dodge") +
  labs(
    title = "Mean Absolute Residuals by Model",
    subtitle = "Residuals increase after the change point for both models",
    x = "",
    y = "Mean Absolute Residual"
  ) +
  theme_classic(base_size = 14) +
  theme(
    plot.title = element_text(face = "bold"),
    plot.subtitle = element_text(size = 11),
    axis.text.x = element_text(size = 11, angle = 15, hjust = 1)
  )

print(p9)

ggsave("figures/model_comparison_mean_residuals.png",
       plot = p9, width = 8, height = 5, dpi = 300)


# -----------------------------
# 12. Sensitivity analysis: different change sizes
# -----------------------------

set.seed(123)

# Change sizes to test
change_sizes <- c(5, 15, 25)

# Empty data frame to store results
sensitivity_results <- data.frame()

for (c_value in change_sizes) {
  
  # Generate new response values
  Y_temp <- rep(NA, n_new)
  
  # Unchanged part
  Y_temp[time <= change_point] <- f_values[time <= change_point] +
    noise[time <= change_point]
  
  # Changed part with different change strength
  Y_temp[time > change_point] <- f_values[time > change_point] +
    c_value * Temp_std[time > change_point]^2 +
    noise[time > change_point]
  
  # Create temporary dataset
  temp_data <- data.frame(
    time = time,
    Y = Y_temp,
    X_new
  )
  
  rownames(temp_data) <- NULL
  
  # Period variable
  temp_data$period <- ifelse(temp_data$time <= change_point, "Before", "After")
  temp_data$period <- factor(temp_data$period, levels = c("Before", "After"))
  
  # Train linear regression on unchanged part
  temp_train <- temp_data[1:change_point, ]
  
  temp_lm <- lm(Y ~ Solar.R + Wind + Temp + Month + Day, data = temp_train)
  
  # Predict on all data
  temp_data$pred_lm <- predict(temp_lm, newdata = temp_data)
  temp_data$abs_resid_lm <- abs(temp_data$Y - temp_data$pred_lm)
  
  # Train random forest on unchanged part
  temp_rf <- randomForest(
    Y ~ Solar.R + Wind + Temp + Month + Day,
    data = temp_train,
    ntree = 300
  )
  
  # Predict on all data
  temp_data$pred_rf <- predict(temp_rf, newdata = temp_data)
  temp_data$abs_resid_rf <- abs(temp_data$Y - temp_data$pred_rf)
  
  # Store linear regression results
  sensitivity_results <- rbind(
    sensitivity_results,
    data.frame(
      Change_Size = c_value,
      Model = "Linear Regression",
      Before_Mean = mean(temp_data$abs_resid_lm[temp_data$period == "Before"]),
      After_Mean = mean(temp_data$abs_resid_lm[temp_data$period == "After"])
    )
  )
  
  # Store random forest results
  sensitivity_results <- rbind(
    sensitivity_results,
    data.frame(
      Change_Size = c_value,
      Model = "Random Forest",
      Before_Mean = mean(temp_data$abs_resid_rf[temp_data$period == "Before"]),
      After_Mean = mean(temp_data$abs_resid_rf[temp_data$period == "After"])
    )
  )
}

# Add increase column
sensitivity_results$Increase <- sensitivity_results$After_Mean -
  sensitivity_results$Before_Mean

print(sensitivity_results)


# -----------------------------
# 13. Plot sensitivity analysis
# -----------------------------

p10 <- ggplot(sensitivity_results,
              aes(x = Change_Size, y = Increase, color = Model)) +
  geom_point(size = 3) +
  geom_line(linewidth = 1) +
  labs(
    title = "Sensitivity to Change Size",
    subtitle = "Larger changes produce larger increases in residuals",
    x = "Change Size c in g(X) = f(X) + c(Temp_std)^2",
    y = "Increase in Mean Absolute Residual"
  ) +
  theme_classic(base_size = 14) +
  theme(
    plot.title = element_text(face = "bold"),
    plot.subtitle = element_text(size = 11)
  )

print(p10)

ggsave("figures/sensitivity_change_size.png",
       plot = p10, width = 8, height = 5, dpi = 300)


# Future work
new_data$residual <- new_data$Ozone - new_data$predicted_ozone
new_data$abs_residual <- abs(new_data$residual)

baseline_mean <- mean(train_data$abs_resid_lm)
baseline_sd <- sd(train_data$abs_resid_lm)

ggplot(new_data, aes(x = time, y = abs_residual)) +
  geom_point() +
  geom_smooth(se = FALSE) +
  labs(
    title = "Future Residuals Over Time",
    x = "Time",
    y = "Absolute Residual"
  )
