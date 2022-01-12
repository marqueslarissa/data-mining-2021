source("libraries.R")
source("model.R")
source("model-prediction.R")
source("utilities.R")

exec_all_horizons <- function(h=HORIZONS,s,runId,training_fn,validation_fn)
{
   results <- NULL
   for(i in 1:length(h))
   {
      results <- rbind(results, main(horizon=h[i],sel_attributes=s,runId,training_fn,validation_fn))
   }
   return(results)
}

# ----- MAIN FUNCTION ------

#
# essa vers?o executa apenas um horizonte
#
main <- function(horizon=selected_horizon, sel_attributes=selected_attributes, runId, training_fn,validation_fn)
{
   require(plyr) # before Hmisc, dplyr
   require(Hmisc) # before dplyr, fields
   require(tidyr) # before sets
   require(sets) # before dplyr, data.table
   require(dplyr) # pbefore data.table
   require(data.table)
   generate_log(" ************* Initiating Execution (LM Version) *************", 1)
   generate_log(EXECUTION_DESCRIPTION)
   generate_log(paste(" REMOVE_LONGER_CASES == ",REMOVE_LONGER_CASES))
   if ( SEARCH_SIMILAR )
      generate_log(paste(" SIMILARITY FOR HANDLING NON-FITTING: set=", SIM_METHOD_SET, "mset=", SIM_METHOD_MSET, "seq=", SIM_METHOD_SEQ))
   else
      generate_log(paste(" SEARCH FOR SIMILARITY = ", SEARCH_SIMILAR, " (searching only exact matches)"))
   generate_log(paste("Training file: ",training_fn,"Validation file:",validation_fn))
   
   startTime <- Sys.time()
   
   # the base name for the files that will store all the results
   statsFile <- paste(EXECUTION_FILE_PREFIX,"_",runId,"_h",horizon,sep="")
   
   model <- NULL
   predict <- NULL
   eval_stats_arr <- NULL
   
   trainingFold <- NULL
   if (EXECUTION_SIZE == Inf)
      trainingFold <- read.csv(file=file.path("../sublogs/grupo4/preprocessed", training_fn)) 
   else
      trainingFold <- read.csv(file=file.path("../sublogs/grupo4/preprocessed", training_fn),nrows = EXECUTION_SIZE)
   
   
   #option to remove from the training the outliers with elapsed time much bigger
   if ( REMOVE_LONGER_CASES == TRUE ) {
      q <- quantile(trainingFold$elapsed_stc,0.99)
      onePerc <- trainingFold[trainingFold$elapsed_stc > q,c("number","elapsed_stc")]
      onePercDist <- distinct(onePerc,onePerc$number)
      colnames(onePercDist) <- c("number")
      generate_log(paste("Removing ",nrow(onePercDist)," traces that have elapsed times bigger than [",q,"] seconds"))
      '%ni%' <- Negate('%in%')
      trainingFold <- trainingFold[trainingFold$number %ni% onePercDist$number,]
   }
   
   rfn <- file.path("../results/pred",paste(statsFile,"_pred.csv",sep=""))
   
   # builds the transition system
   model <- build_ats(trainingFold,horizon,sel_attributes)
   
   # anotates the transition system
   training_stats <- annotate_model(trainingFold, rfn, "T", 0, horizon, model)
   
   if ( is.null(validation_fn) ) {
      
      eval_stats_arr <- rbind(training_stats)
      
   } else {
   
      
      # prediction over the validation data set
      testingFold <- NULL
      if (EXECUTION_SIZE == Inf)
         testingFold <- read.csv(file=file.path("../sublogs/grupo4/preprocessed", validation_fn)) 
      else
         testingFold <- read.csv(file=file.path("../sublogs/grupo4/preprocessed", validation_fn),nrows = EXECUTION_SIZE)
      
      
      # builds the transition system for teh testing fold
      predict <- build_prediction_optimized(testingFold,model)
      validation_stats <- annotate_model(testingFold, rfn, "V", 0, horizon, model)
      
      eval_stats_arr <- rbind(training_stats, validation_stats)
      
      
   }
   
   eval_stats_df1 <- data.frame(eval_stats_arr)
   
   string_attrib <- paste(unlist(sel_attributes),collapse=",")
   eval_stats_df1 <- cbind(start=format(startTime, "%Y-%m-%d %H:%M:%S"),end=format(Sys.time(), "%Y-%m-%d %H:%M:%S"),
                           at=EXECUTION_LOCATION, attributes=string_attrib,eval_stats_df1)
   
   sfilen <- file.path("../results/stats",paste(statsFile,"_STATS.csv",sep=""))
   write.table(eval_stats_df1, file=sfilen, row.names=FALSE, col.names = TRUE, sep=";", dec=",")
   
   generate_log(paste("Stat file generated: [",statsFile,"_STATS.csv]",sep=""))      
   
   generate_log("Step completed successfully.",2)
   
   eval.parent(substitute(LAST_RUN_MODEL<-model))
   eval.parent(substitute(LAST_RUN_PREDICT<-predict))
   
   return(eval_stats_df1)
   
}


# objetivo: avaliar solu??es para tratar non-fitting

# make sure that all the important constants are set before start
TRACE_ID_COLUMN_NAME <- "number"
STATUS_COLUMN_NAME <- "incident_state"
EVENT_TIMESTAMP_COLUMN_NAME <- "updated_at"
CLOSED_STATUS_VALUE <- "closed"
REMOVE_LONGER_CASES <- FALSE

SEARCH_SIMILAR <- FALSE
# possible methods are dl, jaccard
SIM_METHOD_SET <- "jaccard" 
SIM_METHOD_MSET <- "jaccard"
SIM_METHOD_SEQ <- "dl"

SAVE_MODEL <- FALSE
LAST_RUN_MODEL <- NULL
LAST_RUN_PREDICT <- NULL

EXECUTION_DESCRIPTION <- "Rabobank Log - Grupo 4 2022"
   EXECUTION_LOCATION <- "pc-lari"
EXECUTION_FILE_PREFIX <- "Jan22"
EXECUTION_SIZE <- Inf # 104

HORIZONS <- c(1,3,5,6,7,Inf)

# these were the best horizons selected according to the tests using the
# incident_state, category and priority (expert selection)

selected_horizon <- c(Inf)
selected_attributes <- c("incident_state"
                         , "assignment_group"
                         , "km_number"
                         ) # expert



# lista de arquivos j? carregados no pr?-proc
# pre-proc foi alterado nessa execu??o para n?o renomear os arquivos
files <- list.files("../sublogs/grupo4/preprocessed")
print(files)
print(files[1])
print(files[2])
tamanho <- length(files)
print(paste("Quantidade de arquivos a serem processados: ",tamanho))

#for (k in 1:3) {
#   cat(paste("=========== Parte",k,"==========="))
#   cat("","","",sep="\n")
   for (j in 1:tamanho-1)
   {
      if (j %% 2 == 0) 
      {
         filen <- files[j+1]
         filenn <- files[j+2]
         id <- gsub(".csv", "", files[j+1])
         
         cat(paste("======== id: ", j, ", file:",filen,"========"))
         cat("","",sep="\n")
         set_log_file(paste("../results/logs/log_",id,".txt",sep=""))
         r <- exec_all_horizons(h=selected_horizon,selected_attributes,id,filen,filenn)
         close_log_file()
         Sys.sleep(time = 60) # needed to make sure that the log file is distinct from the previous one
      }
   }
#   id_orig <- id_orig + 100
#}