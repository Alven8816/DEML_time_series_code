library(caret)
library(deeper)
library(caret)
library(xgboost)
library(forecast)

# load example data

mydata_final_clean <- read.csv(file = "data_example.csv")

mydata_final_clean$X <- NULL # delete useless 'X' variable


# create a function to conduct DEML time-series framework

### we select 3 base model in the function. For other models such as LSTM, users
### can training it independently and combined the prediction into the function.
DEML_model <- function(data = sensor_single_df,
                                  test_period = 7,
                                  horizen = days(1)) {
  ## set last 7 days as final unseen testing dataset
  split_date = lubridate::ymd(str_extract(
    string = as.character(max(data$time_utc)), pattern = "[0-9-]*"
  ))- days(test_period)
  # set the initial training dataset
  trainset <-
    data[data$time_utc <= split_date, ]
  print(dim(trainset))
  #====================
  pred_rf = list()
  pred_xgb = list()
  pred_svm = list()
  
  pred_train_rf = list()
  pred_train_xgb = list()
  pred_train_svm = list()
  
  # for base model combination
  test_pred = list()
  train_pred = list()
  # meta predict results
  pred_train_rf_meta = list()
  pred_train_glm_meta = list()
  
  pred_rf_meta = list()
  pred_glm_meta = list()
  # for meta model combination
  test_pred_meta = list()
  train_pred_meta = list()
  # for deml
  deml_train = list()
  deml_test = list()
  
  all_train = list()
  all_test = list()
  #=======
  #testset <- data.frame()
  
  # here we have a loop of 7 days
  for (i in 1:(days(test_period) / horizen)) {
    # set the initial date for testing dadaset
    test_date <-
      split_date + horizen*(i)
    print(test_date)
    # if the test_date exceed the final date in the testing dataset, the loop break
    if (test_date > ymd(str_extract(string = as.character(max(data$time_utc)), pattern = "[0-9-]*"))) {
      break
    } else {
      testset <- data[data$time_utc > test_date &
                        data$time_utc <= (test_date + horizen), ]
      print(dim(testset))
    }
    if (nrow(testset)==0) { # here we skip the situation where the testset is empty
      next
    }
    #================================
    #fit the model
    model_rf <-
      ranger::ranger(
        temperature_indoor_impute2 ~ .,
        data = trainset[, colnames(trainset) %in% c("time_utc", "LocationID", "Device.ID") == FALSE], # exclude 3 variables in the dataset
        mtry = 10,
        num.trees = 50,
        importance = "permutation",
        seed = 2020
        #quantreg = TRUE # for QRF
        #keep.inbag = TRUE
      )
    # predict using the train set
    pred_train_rf[[i]] <-
      stats::predict(model_rf, trainset)$predictions
    
    # predict using the test set
    pred_rf[[i]] <-
      stats::predict(model_rf, testset)$predictions
    
    
    #================================
    # XGBoost
    ##seperate the train/test subset
    train_xgb <-
      data.matrix(trainset[, colnames(trainset) %in% c("time_utc",
                                                       "LocationID",
                                                       "Device.ID",
                                                       "temperature_indoor_impute2") == FALSE])
    train_label <-
      data.matrix(trainset[, "temperature_indoor_impute2"])
    
    test_xgb <-
      data.matrix(testset[, colnames(trainset) %in% c("time_utc",
                                                      "LocationID",
                                                      "Device.ID",
                                                      "temperature_indoor_impute2") == FALSE])
    test_label <-
      data.matrix(testset[, "temperature_indoor_impute2"])
    
    # training the model
    bst <-
      xgboost::xgboost(
        data = train_xgb,
        label = train_label,
        max.depth = 10,
        eta = 0.3,
        nround = 10,
        nthread = 4,
        verbose = 0,
        objective = 'reg:squarederror'
      )
    # predict using the train set
    pred_train_xgb[[i]] <-
      data.frame(xgb_pred = stats::predict(bst, train_xgb))
    # model prediction
    pred_xgb[[i]] <-
      data.frame(xgb_pred = stats::predict(bst, test_xgb))
    #================================
    
    fit_svm <-
      e1071::svm(temperature_indoor_impute2 ~ .,
                 data = trainset[, colnames(trainset) %in% c("time_utc",
                                                             "LocationID",
                                                             "Device.ID") ==
                                   FALSE],
                 cross = 0,
                 kernel = 'radial')
    # predict using the train set
    pred_train_svm[[i]] <- stats::predict(fit_svm, trainset)
    # prediction SVR
    pred_svm[[i]] <- stats::predict(fit_svm, testset)
    
    # combine the train prediction
    train_pred[[i]] <-
      cbind(
        trainset[, colnames(trainset) %in% c("time_utc", "LocationID", "Device.ID") == FALSE],
        rf_pred = pred_train_rf[[i]],
        xgb_pred = pred_train_xgb[[i]],
        svm_pred = pred_train_svm[[i]]
      )
    ## other model results such as LSTM results for training can be combined here 
    
    # combine the test prediction
    test_pred[[i]] <-
      cbind(
        testset[, colnames(trainset) %in% c("time_utc", "LocationID", "Device.ID") == FALSE],
        rf_pred = pred_rf[[i]],
        xgb_pred = pred_xgb[[i]],
        svm_pred = pred_svm[[i]]
      )
    ## other model results such as LSTM results for testing can be combined here
    
    ########################### META #################
    ## training meta-models: RF, GLM
    #fit the RF meta model
    #================================
    model_rf_meta <-
      ranger::ranger(
        temperature_indoor_impute2 ~ .,
        data = train_pred[[i]],
        mtry = 10,
        num.trees = 50,
        importance = "permutation",
        seed = 2020
      )
    # predict using the train set
    pred_train_rf_meta[[i]] <-
      stats::predict(model_rf_meta, train_pred[[i]])$predictions
    
    # predict using the test set
    pred_rf_meta[[i]] <-
      stats::predict(model_rf_meta, test_pred[[i]])$predictions
    
    #fit the GLM meta model
    model_glm_meta <-
      glm(
        formula =  temperature_indoor_impute2 ~ rf_pred + xgb_pred + svm_pred,
        family = gaussian,
        data = train_pred[[i]]
      )
    
    # predict using the train set
    pred_train_glm_meta[[i]] <-
      stats::predict(model_glm_meta, train_pred[[i]], type = "response")
    
    # predict using the test set
    pred_glm_meta[[i]] <-
      stats::predict(model_glm_meta, test_pred[[i]])
    
    # combine meta model results
    train_pred_meta[[i]] <-
      as.matrix(x = data.table(pred_rf_meta = pred_train_rf_meta[[i]],
                               pred_glm_meta = pred_train_glm_meta[[i]]))
    
    test_pred_meta[[i]] <-
      as.matrix(x = data.table(pred_rf_meta = pred_rf_meta[[i]],
                               pred_glm_meta = pred_glm_meta[[i]]))
    
    # using NNLS to obtain the weights of meta models
    y = as.matrix(train_pred[[i]][, "temperature_indoor_impute2"])

    x = as.matrix(x = data.table(train_pred_meta[[i]]))
    nnls_weight <- nnls::nnls(A = x, b = y)
    print(paste0(i, " weights:", nnls_weight$x))
    deml_train[[i]] <-
      data.frame(deml_train = x %*% t(matrix(nnls_weight$x, ncol = length(nnls_weight$x))))
    
    # nnls for testing data
    x_new = test_pred_meta[[i]]
    
    deml_test[[i]] <-
      data.frame(deml_test = x_new %*% t(matrix(nnls_weight$x, ncol = length(nnls_weight$x))))
    
    all_train[[i]] <-
      data.frame(
        data.table(
          rf_pred = pred_train_rf[[i]],
          xgb_pred = pred_train_xgb[[i]],
          svm_pred = pred_train_svm[[i]],
          train_pred_meta[[i]],
          deml_train[[i]]
        )
      )
    
    all_test[[i]] <- data.frame(
      data.table(
        rf_pred = pred_rf[[i]],
        xgb_pred = pred_xgb[[i]],
        svm_pred = pred_svm[[i]],
        test_pred_meta[[i]],
        deml_test[[i]]
      )
    ) %>%
      bind_cols(testset[, c("LocationID",
                            "Device.ID",
                            "time_utc",
                            "temperature_indoor_impute2")])
    # #================================
    trainset <- rbind(trainset, testset) # we extend the testset into traning set for next loop
    #print(dim(trainset))
  }
  all_test_df <- bind_rows(all_test)
  return(all_test_df)
}

########################### DEML model training for each sensor ###############

#all_model_deml_result <- list()
e1 = Sys.time()#
for (j in unique(mydata_final_clean$Device.ID)) {
  print(j)
  sensor_single_df <- mydata_final_clean %>%
    filter(Device.ID %in% c(j))
  
  # select the last 7 days as testing dataset
  all_model_deml_result <-
    DEML_model(data = sensor_single_df,
                          test_period = 7,
                          horizen = days(1))
  saveRDS(all_model_deml_result, 
          file = paste0("./sensor_",j,"_model_deml_result.rds"))
}
e2 = Sys.time()
e2-e1

