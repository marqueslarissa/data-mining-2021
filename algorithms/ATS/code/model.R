
NIL_STATE <- -1


#' funcao de construcao do MTA, recebe o trace, lista de instancias
#' Código fonte original
#'
#' @param events : list of events
#' @param horiz : the horizon
#' @param sel_attributes : list of selected attributes
#'
#' @return
#' @export
#'
#' @examples
build_ats <- function(aevents, horiz, sel_attributes)
{
   
   # mudanças nos nomes das variáveis
   #  nome no código original    novo nome
   #  asel_traces_lcl            events
   #  sel_traces_lcl             events (copy)
   #  process_instances          traces
   #  trace                      traceEvents
   
   events <- aevents  # necessário para atualizar a lista original no final da execução
   
   generate_log(paste("Starting build_ats for horizon",horiz,"and attributes [",paste( unlist(sel_attributes), collapse=', '),"]"),2)
   
   traces <- distinct_(events, TRACE_ID_COLUMN_NAME)
   
   traces_states <-list(
      number=traces,
      set_states=list(),
      mset_states=list(),
      seq_states=list()
   )
   
   # listas com os estados de cada modelo
   seq_state_list <- list()
   set_state_list <- list()
   mset_state_list <- list()
   
   # will store the values as separated fields and not only as a character representation.
   # This is necessary to rebuild the search string in the prediction phase
   # Also, it is necessary to keep the three lists as each abstration will generate a
   # distinct list (except for horizon 1)
   full_sequence <- list()
   full_set <- list()
   full_mset <- list()
   
   generate_log("   Starting loops over all traces",2)
   
   for(i in 1:nrow(traces))
   {
      
      #print(paste("inicio loop principal: ",i))
      
      # busca eventos do caso / search events of the given trace
      traceEvents <- events[events$number == traces$number[i],]
      
      #generate_log(paste("          trace",traces$number[i],"with",nrow(traceEvents),"events"),2)
      
      # it must happen due to the backward update to the original events in the end of this function
      traceEvents$seq_state_id <- 0
      traceEvents$set_state_id <- 0
      traceEvents$mset_state_id <- 0
      traceEvents$sojourn_set_stc <- 0
      traceEvents$sojourn_mset_stc <- 0
      traceEvents$sojourn_seq_stc <- 0
      
      activity_fields <- do.call(paste, as.data.frame(traceEvents[,sel_attributes]))
      
      # vector with the timestamps for when the event has started
      timestamp_field <-  as.vector(traceEvents[, EVENT_TIMESTAMP_COLUMN_NAME])
      
      # Step 1: populate the lists of states for each abstraction
      seq_list <- list()
      set_list <- list()
      mset_list <- list()
      for (j in 1:nrow(traceEvents))
      {
         #print(paste("inicio loop 1: ",j))
         
         # calculate based on the horizon
         horiz_index <- max(j - horiz + 1, 1)
         
         # generates state for SEQUENCE abstration
         inc_seq <- as.character(activity_fields[horiz_index:j])
         seq_list[[j]]  <- toString(inc_seq)
         
         stateId <- match(seq_list[[j]],seq_state_list,NIL_STATE)
         if ( stateId == NIL_STATE ) {
            seq_state_list <- append(seq_state_list, seq_list[[j]])
            #full_sequence <- append(full_sequence, list(activity_fields[horiz_index:j]))
            full_sequence <- append(full_sequence, list(inc_seq))
            stateId <- length(seq_state_list)
         }
         traceEvents[j,"seq_state_id"] <- stateId
         
         
         # generates state for SET
         inc_set <- as.set(inc_seq)
         set_list[[j]]  <- toString(inc_set)
         
         # generates state for multi-set
         inc_gset <- as.gset(inc_seq)
         inc_gset_str <- toString(rbind(gset_support(inc_gset), gset_memberships(inc_gset)))
         mset_list[[j]]  <- inc_gset_str
         
         if (horiz==1) # horizon == 1, all will be the same
         {
            traceEvents[j,c("set_state_id","mset_state_id","seq_state_id")] <- stateId
         }
         else
         {
            # if the horizon is > 1 the abstractions will generate distinct lists
            stateId <- match(set_list[[j]],set_state_list,NIL_STATE)
            if ( stateId == NIL_STATE ) {
               set_state_list <- append(set_state_list, set_list[[j]])
               # stores the sequence list, as from it will be possible to extract the set
               full_set <- append(full_set, list(inc_seq))
               stateId <- length(set_state_list)
            }
            traceEvents[j,"set_state_id"] <- stateId
            
            # now for the multi-set
            stateId <- match(mset_list[[j]],mset_state_list,NIL_STATE)
            if ( stateId == NIL_STATE ) {
               mset_state_list <- append(mset_state_list, mset_list[[j]])
               # stores the sequence list, as from it will be possible to extract the mset
               full_mset <- append(full_mset, list(inc_seq))
               stateId <- length(mset_state_list)
            }
            traceEvents[j,"mset_state_id"] <- stateId
         } # END IF horizon==1
      } # END LOOP over trace events
      
      #if (horiz==1) # all the same, just need to copy the list built for the sequence
      #   full_set <- full_mset <- full_sequence
      
      # Step 2, calculate the sojourn
      calculate_sojourn_v2(traceEvents)
      
      # armazena resultado das transicoes de estado para instancia atual
      # modelo de abstracao sequencia
      traces_states$seq_states[i] <- list(traceEvents[,"seq_state_id"])
      
      # modelo de abstracao set
      traces_states$set_states[i] <- list(traceEvents[,"set_state_id"])
      
      # modelo de abstracao mset
      traces_states$mset_states[i] <- list(traceEvents[,"mset_state_id"])
      
      # guardo a estado no evento
      events[events$number == traces$number[i],]$seq_state_id <- traceEvents$seq_state_id
      events[events$number == traces$number[i],]$set_state_id <- traceEvents$set_state_id
      events[events$number == traces$number[i],]$mset_state_id <- traceEvents$mset_state_id
      events[events$number == traces$number[i],]$sojourn_set_stc <- traceEvents$sojourn_set_stc
      events[events$number == traces$number[i],]$sojourn_mset_stc <- traceEvents$sojourn_mset_stc
      events[events$number == traces$number[i],]$sojourn_seq_stc <- traceEvents$sojourn_seq_stc
      
   } # fim i
   
   generate_log("   Ending loops over all traces",2)
   
   # initializes these attributes - they will not be used here, but will be necessary for the prediction
   events$set_nf <- 0
   events$mset_nf <- 0
   events$seq_nf <- 0
   
   # retorna resultado - precisa atualizar a lista de eventos original, para os cálculos
   eval.parent(substitute(aevents<-events))
   
   generate_log("   Summarizing the values for the model",2)
   
   # filtrar os valores que sao estados finais pois distorcem a media
   # Alexandre: version exp3 NF:
   # removed this filter to avoid states not present in the summary because they were always the final states
   # in the training - but it does not mean that they would not appear somewhere else in the validation data.
   #events_anot_filtered  <- events[events$remaining_stc > 0,]
   events_anot_filtered <- events
   
   summary_set <- gen_summary_pred_fn(events_anot_filtered, 'set_state_id','remaining_stc')
   summary_mset <- gen_summary_pred_fn(events_anot_filtered, 'mset_state_id','remaining_stc')
   summary_seq <- gen_summary_pred_fn(events_anot_filtered, 'seq_state_id','remaining_stc')
   
   summary_sj_set <- gen_summary_pred_fn(events_anot_filtered, 'set_state_id','sojourn_set_stc')
   summary_sj_mset <- gen_summary_pred_fn(events_anot_filtered, 'mset_state_id','sojourn_mset_stc')
   summary_sj_seq <- gen_summary_pred_fn(events_anot_filtered, 'seq_state_id','sojourn_seq_stc')
   
   
   mta_model <- list(traces_states=traces_states,
                     seq_state_list=seq_state_list,
                     set_state_list=set_state_list,
                     mset_state_list=mset_state_list,
                     full_set=full_set, full_mset=full_mset, full_sequence=full_sequence,
                     summary_set=summary_set, summary_mset=summary_mset, summary_seq=summary_seq,
                     summary_sj_set=summary_sj_set, summary_sj_mset=summary_sj_mset, summary_sj_seq=summary_sj_seq,
                     horiz=horiz,
                     sel_attributes=sel_attributes
   )
   
   generate_log("Ended build_ats",2)
   
   return(mta_model)
}


#' Cálculo de Sojourn usado na primeira execução completa
#' Não é o código original - foi modificado para seguir mais na linha do que foi
#' proposto pelo van der Aalst no artigo de Time Prediction
#'
#' @param traceEvents
#'
#' @return
#' @export
#'
#' @examples
calculate_sojourn_v1 <- function(aTraceEvents) {
   
   traceEvents <- aTraceEvents
   
   curr_sojourn_set_state <- traceEvents[1,"set_state_id"]
   curr_sojourn_mset_state <- traceEvents[1,"mset_state_id"]
   curr_sojourn_seq_state <- traceEvents[1,"seq_state_id"]
   curr_sojourn_set_stc <- 0
   curr_sojourn_mset_stc <- 0
   curr_sojourn_seq_stc <- 0
   # first event of each case, the sojourn will be always zero
   for (j in 2:nrow(traceEvents))
   {
      elapsed <- traceEvents[j,"elapsed_stc"]
      # advances until it reaches a distinct state - for SET
      if ( traceEvents[j,"set_state_id"] != curr_sojourn_set_state ) {
         curr_sojourn_set_state <- traceEvents[j,"set_state_id"]
         traceEvents[j,"sojourn_set_stc"] <- elapsed - curr_sojourn_set_stc
         curr_sojourn_set_stc <- elapsed
      }
      
      # now for MULTI-SET
      if ( traceEvents[j,"mset_state_id"] != curr_sojourn_mset_state ) {
         curr_sojourn_mset_state <- traceEvents[j,"mset_state_id"]
         traceEvents[j,"sojourn_mset_stc"] <- elapsed - curr_sojourn_mset_stc
         curr_sojourn_mset_stc <- elapsed
      }
      
      # and now for SEQUENCE
      if ( traceEvents[j,"seq_state_id"] != curr_sojourn_seq_state ) {
         curr_sojourn_seq_state <- traceEvents[j,"seq_state_id"]
         traceEvents[j,"sojourn_seq_stc"] <- elapsed - curr_sojourn_seq_stc
         curr_sojourn_seq_stc <- elapsed
      }
      
   } # fim j
   
   eval.parent(substitute(aTraceEvents<-traceEvents))
   
}



#' Cálculo de Sojourn versão 2
#' (na verdade, versão 3 se considerar a 1a como sendo a do código original)
#' Neste caso a alteração é que o resultado do sojourn calculado deverá atualizar a
#' célula j-1 e não a j.
#'
#' @param traceEvents
#'
#' @return
#' @export
#'
#' @examples
calculate_sojourn_v2 <- function(aTraceEvents) {
   
   traceEvents <- aTraceEvents
   
   # gets the first state for each abstraction, as the current
   curr_sojourn_set_state <- traceEvents[1,"set_state_id"]
   curr_sojourn_mset_state <- traceEvents[1,"mset_state_id"]
   curr_sojourn_seq_state <- traceEvents[1,"seq_state_id"]
   curr_sojourn_set_stc <- 0
   curr_sojourn_mset_stc <- 0
   curr_sojourn_seq_stc <- 0
   # jump to second event as the update will be on j-1
   for (j in 2:nrow(traceEvents))
   {
      elapsed <- traceEvents[j,"elapsed_stc"]
      # advances until it reaches a distinct state - for SET
      if ( traceEvents[j,"set_state_id"] != curr_sojourn_set_state ) {
         curr_sojourn_set_state <- traceEvents[j,"set_state_id"]
         traceEvents[j-1,"sojourn_set_stc"] <- elapsed - curr_sojourn_set_stc
         curr_sojourn_set_stc <- elapsed
      }
      
      # now for MULTI-SET
      if ( traceEvents[j,"mset_state_id"] != curr_sojourn_mset_state ) {
         curr_sojourn_mset_state <- traceEvents[j,"mset_state_id"]
         traceEvents[j-1,"sojourn_mset_stc"] <- elapsed - curr_sojourn_mset_stc
         curr_sojourn_mset_stc <- elapsed
      }
      
      # and now for SEQUENCE
      if ( traceEvents[j,"seq_state_id"] != curr_sojourn_seq_state ) {
         curr_sojourn_seq_state <- traceEvents[j,"seq_state_id"]
         traceEvents[j-1,"sojourn_seq_stc"] <- elapsed - curr_sojourn_seq_stc
         curr_sojourn_seq_stc <- elapsed
      }
      
   } # fim j
   
   eval.parent(substitute(aTraceEvents<-traceEvents))
   
}

#' funcao de construcao dos MTA recebe o trace, lista de instancias
#'
#' @param asel_traces_lcl
#' @param process_instances
#' @param ats
#'
#' @return
#' @export
#'
#' @examples
#build_prediction <- function(asel_traces_lcl, process_instances, mta_in)
build_prediction <- function(aevents, ats)
{
   generate_log("Starting build_prediction",2)
   
   events <- aevents  # necessário para atualizar a lista original no final da execução
   
   traces <- distinct_(events, TRACE_ID_COLUMN_NAME)
   
   #   process_instances_states <-list(
   traces_states <-list(
      number=traces,
      set_states=list(),
      mset_states=list(),
      seq_states=list()
   )
   
   # listas com os estados de cada modelo
   seq_state_list <- ats$seq_state_list
   set_state_list <- ats$set_state_list
   mset_state_list <- ats$mset_state_list
   
   # initializes the flags to indicate if the event is going for a non-fitting path
   events$set_nf <- 0
   events$mset_nf <- 0
   events$seq_nf <- 0
      
   for(i in 1:nrow(traces))
   {
      #generate_log(paste("trace ",i,traces$number[i]),2)
      
      # busca eventos do caso / search events of the given trace
      traceEvents <- events[events$number == traces$number[i],]
      
      traceEvents$seq_state_id <- 0
      traceEvents$set_state_id <- 0
      traceEvents$mset_state_id <- 0
      traceEvents$sojourn_set_stc <- 0
      traceEvents$sojourn_mset_stc <- 0
      traceEvents$sojourn_seq_stc <- 0
      traceEvents$set_nf <- 0
      traceEvents$mset_nf <- 0
      traceEvents$seq_nf <- 0
      
      # label utilizado na funcao de eventos (campos do modelo)
      activity_fields <- do.call(paste, as.data.frame(traceEvents[,ats$sel_attributes]))
      
      # tempo em que ocorreu o evento
      timestamp_field <-  as.vector(traceEvents[, EVENT_TIMESTAMP_COLUMN_NAME])
      
      # listas com os modelos de abstracao usados para representacao dos eventos
      seq_list <- list()
      set_list <- list()
      mset_list <- list()
      seq_in_non_fit_path <- set_in_non_fit_path <- mset_in_non_fit_path <- FALSE
      similar_seq_id <- similar_set_id <- similar_mset_id <- NIL_STATE
      for (j in 1:nrow(traceEvents))
      {
         #generate_log(paste("event ",j),2)
         
         # calculate the horizon
         horiz_index <- max(j - ats$horiz + 1, 1)
         
         # generates state for SEQUENCE abstration
         inc_seq <- as.character(activity_fields[horiz_index:j])
         # to handle the non-fitting path 
         inc_seq_nf <- as.character((activity_fields[j]))
         
         if (seq_in_non_fit_path) {
            new_value <- ats$full_sequence[[similar_seq_id]]
            new_value <- append(new_value,inc_seq_nf)
            seq_list[[j]]  <- toString(new_value)
         } else {
            seq_list[[j]]  <- toString(inc_seq)
         }
         
         # searches for the exact match
         stateId <- match(seq_list[[j]], seq_state_list, NIL_STATE)
         
         if (ats$horiz==1) # horizonte 1, todos iguais
         {
            if (SEARCH_SIMILAR) {
              seq_in_non_fit_path <- set_in_non_fit_path <- mset_in_non_fit_path <- (stateId == NIL_STATE)
              if (seq_in_non_fit_path) {
                 traceEvents[j,c("set_nf","mset_nf","seq_nf")] <- 1
                 stateId <- which.max(stringsim(a=seq_list[[j]], b=unlist(seq_state_list), method=SIM_METHOD_SEQ, useBytes=FALSE))
                 similar_seq_id <- similar_set_id <- similar_mset_id <- stateId
              }
            }
            traceEvents[j,c("set_state_id","mset_state_id","seq_state_id")] <- stateId
         } else {
            if (SEARCH_SIMILAR) {
               seq_in_non_fit_path <- ( stateId == NIL_STATE )
               if(seq_in_non_fit_path) {
                  traceEvents[j,"seq_nf"] <- 1
                  stateId <- which.max(stringsim(a=seq_list[[j]], b=unlist(seq_state_list), method=SIM_METHOD_SEQ, useBytes=FALSE))
                  similar_seq_id <- stateId
               }
            }
            traceEvents[j,"seq_state_id"] <- stateId
            
            # generates state for SET
            if (set_in_non_fit_path) {
               inc_set <- ats$full_set[[similar_set_id]]
               inc_set <- append(inc_set,inc_seq_nf)
            } else {
               inc_set <- as.set(inc_seq)
            }
            set_list[[j]]  <- toString(inc_set)
            
            # executes the exact search
            stateId <- match(set_list[[j]], set_state_list, NIL_STATE)
            
            if (SEARCH_SIMILAR) {
               set_in_non_fit_path <- ( stateId == NIL_STATE )
               if (set_in_non_fit_path) {
                  # handling the non-fitting - using the DL distance to get the most similar entry
                  traceEvents[j,"set_nf"] <- 1
                  stateId <- which.max(stringsim(a=set_list[[j]], b=unlist(set_state_list), method=SIM_METHOD_SET, useBytes=FALSE))
                  similar_set_id <- stateId
               }
            }
            traceEvents[j,"set_state_id"] <- stateId
            
            # multi-set (requires package sets)
            if (mset_in_non_fit_path) {
               inc_gset <- ats$full_mset[[similar_mset_id]]
               inc_gset <- append(inc_gset,inc_seq_nf)
               inc_gset <- as.gset(inc_gset)
            } else {
               inc_gset <- as.gset(inc_seq)
            }
            
            inc_gset_str <- toString(rbind(gset_support(inc_gset), gset_memberships(inc_gset)))
            mset_list[[j]]  <- inc_gset_str            
            
            # executes the exact search
            stateId <- match(mset_list[[j]], mset_state_list, NIL_STATE)
            
            if (SEARCH_SIMILAR) {
               mset_in_non_fit_path <- ( stateId == NIL_STATE )
               if(mset_in_non_fit_path) {
                  traceEvents[j,"mset_nf"] <- 1
                  stateId <- which.max(stringsim(a=mset_list[[j]], b=unlist(mset_state_list), method=SIM_METHOD_MSET, useBytes=FALSE))
                  similar_mset_id <- stateId
               }
            }
            traceEvents[j,"mset_state_id"] <- stateId
            
         } # end if horizon == 1
         
      } # end loop over trace events

      # Step 2, calculate the sojourn
      calculate_sojourn_v2(traceEvents)

      # armazena resultado das transicoes de estado para instancia atual
      # modelo de abstracao sequencia
      traces_states$seq_states[i] <- list(traceEvents[,"seq_state_id"])
      
      # modelo de abstracao set
      traces_states$set_states[i] <- list(traceEvents[,"set_state_id"])
      
      # modelo de abstracao mset
      traces_states$mset_states[i] <- list(traceEvents[,"mset_state_id"])
      
      # guardo a estado no evento
      events[events$number == traces$number[i],]$seq_state_id <- traceEvents$seq_state_id
      events[events$number == traces$number[i],]$set_state_id <- traceEvents$set_state_id
      events[events$number == traces$number[i],]$mset_state_id <- traceEvents$mset_state_id
      events[events$number == traces$number[i],]$sojourn_set_stc <- traceEvents$sojourn_set_stc
      events[events$number == traces$number[i],]$sojourn_mset_stc <- traceEvents$sojourn_mset_stc
      events[events$number == traces$number[i],]$sojourn_seq_stc <- traceEvents$sojourn_seq_stc
      events[events$number == traces$number[i],]$set_nf <- traceEvents$set_nf
      events[events$number == traces$number[i],]$mset_nf <- traceEvents$mset_nf
      events[events$number == traces$number[i],]$seq_nf <- traceEvents$seq_nf
      
   } # end loop over all the traces
   
   # retorna resultado - precisa atualizar a lista de eventos original, para os cálculos
   eval.parent(substitute(aevents<-events))
   
   mta_model <- list(traces_states=traces_states,
                     #seq_state_list=ats$seq_state_list,
                     #set_state_list=ats$set_state_list,
                     #mset_state_list=ats$mset_state_list,
                     horiz=ats$horiz,
                     sel_attributes=ats$sel_attributes
   )
   
   generate_log("Ended build_prediction",2)
   
   return(mta_model)
}


#' Title
#'
#' @param data
#' @param groupvars
#' @param measurevar
#' @param na.rm
#' @param conf.interval
#' @param .drop
#'
#' @return
#' @export
#'
#' @examples
#'  summary_set <- gen_summary_pred_fn(incidentevtlog_anot_st, 'set_state_id','remaining_stc')

gen_summary_pred_fn <- function(data=NULL, groupvars=NULL, measurevar,  na.rm=TRUE,
                                conf.interval=.95, .drop=TRUE) {
   
   # New version of length which can handle NA's: if na.rm==T, don't count them
   length2 <- function (x, na.rm=FALSE) {
      if (na.rm) sum(!is.na(x))
      else       length(x)
   }
   
   # This does the summary. For each group's data frame, return a vector with
   # N, mean, and sd
   datac <- ddply(data, groupvars, .drop=.drop,
                  .fun = function(xx, col) {
                     c(N    = length2(xx[[col]], na.rm=na.rm),
                       mean = ceiling(mean   (xx[[col]], na.rm=na.rm)),
                       sd   = ceiling(sd     (xx[[col]], na.rm=na.rm)),
                       median   = ceiling(median (xx[[col]], na.rm=na.rm)),
                       min   = ceiling(min     (xx[[col]], na.rm=na.rm)),
                       max   = ceiling(max     (xx[[col]], na.rm=na.rm))
                     )
                  },
                  measurevar
   )
   
   # Rename the "mean" column
   #datac <- rename(datac, c("mean" = measurevar))
   
   datac$se <- datac$sd / sqrt(datac$N)  # Calculate standard error of the mean
   
   # Confidence interval multiplier for standard error
   # Calculate t-statistic for confidence interval:
   # e.g., if conf.interval is .95, use .975 (above/below), and use df=N-1
   ciMult <- qt(conf.interval/2 + .5, datac$N-1)
   
   datac$ci <- datac$se * ciMult
   
   # registro para valores nao encontrados #non_fitting
   datac <- rbind(datac,
                  c(NIL_STATE, sum(datac$N),  mean(datac$mean),  sd(datac$mean),
                    median(datac$mean), min(datac$mean), max(datac$mean))
   )
   
   return(datac)
}





#' Title
#'
#' @param fold_events
#' @param resultFile
#' @param type
#' @param fold
#' @param horiz
#'
#' @return
#' @export
#'
#' @examples
#'
#' This function was extrated from the original function
#' #eval_model_gen_fn <- function(lsel_traces_list)
#' #eval_model_gen_fn <- function(events)
#'
annotate_model <- function(fold_events, resultFile, type, fold, horiz, model)
{
   
   generate_log("Annotating the model...",2)
   
   summary_pred_stats <- NULL
   result <- NULL
   
   events_anot <- as.data.frame(
      fold_events[, c("number", "updated_at", "incident_state", "seq_state_id","set_state_id", "mset_state_id",
                      "sojourn_set_stc","sojourn_mset_stc","sojourn_seq_stc","elapsed_stc", "remaining_stc",
                      "set_nf", "mset_nf", "seq_nf")])
   
   # set
   events_anot$set_state_remain_stc <-
      model$summary_set$mean[match(events_anot$set_state_id, model$summary_set$set_state_id)]
   events_anot$set_state_sj_stc <-
      model$summary_sj_set$mean[match(events_anot$set_state_id, model$summary_sj_set$set_state_id)]

#   events_anot$set_state_remain_sd <-
#      model$summary_set$sd[match(events_anot$set_state_id, model$summary_set$set_state_id)]
      
   # multi set
   events_anot$mset_state_remain_stc <-
      model$summary_mset$mean[match(events_anot$mset_state_id, model$summary_mset$mset_state_id)]
   events_anot$mset_state_sj_stc <-
      model$summary_sj_mset$mean[match(events_anot$mset_state_id, model$summary_sj_mset$mset_state_id)]

#   events_anot$mset_state_remain_sd <-
#      model$summary_mset$sd[match(events_anot$mset_state_id, model$summary_mset$mset_state_id)]
   
   # sequence
   events_anot$seq_state_remain_stc <-
      model$summary_seq$mean[match(events_anot$seq_state_id, model$summary_seq$seq_state_id)]
   events_anot$seq_state_sj_stc <-
      model$summary_sj_seq$mean[match(events_anot$seq_state_id, model$summary_sj_seq$seq_state_id)]
   
#   events_anot$seq_state_remain_sd <-
#      model$summary_seq$sd[match(events_anot$seq_state_id, model$summary_seq$seq_state_id)]
   
   
   # prediction based only in the mean
   calculate_prediction(events_anot)

   # prediction based in the mean and sojourn
   calculate_prediction_with_sojourn(events_anot)
   

   
   # remove valorers sem match para calculo erro
   # funcao remove todas as linhas que tiverem alguma coluna missing (nula)
   events_anot_filtered <- na.omit(events_anot)
   # remove valores dos estados finais Target = 0 que distorcem a media
   # valores do ultimo estado serao sempre precisos
   events_anot_filtered <- events_anot_filtered[events_anot_filtered$remaining_stc > 0,]
   
   # calculo erro  MAPE p todos os registros
   
   #MAPE(y_pred, y_true)
   mape_val <- c(
      MAPE(events_anot_filtered$predict_remain_set, events_anot_filtered$remaining_stc),
      MAPE(events_anot_filtered$predict_remain_mset, events_anot_filtered$remaining_stc),
      MAPE(events_anot_filtered$predict_remain_seq, events_anot_filtered$remaining_stc)
   )
   names(mape_val) <- c("MAPE_SET","MAPE_MSET","MAPE_SEQ")
   
   #RMSPE(y_pred, y_true)
   rmspe_val <- c(
      RMSPE(events_anot_filtered$predict_remain_set, events_anot_filtered$remaining_stc),
      RMSPE(events_anot_filtered$predict_remain_mset, events_anot_filtered$remaining_stc),
      RMSPE(events_anot_filtered$predict_remain_seq, events_anot_filtered$remaining_stc)
   )
   names(rmspe_val) <- c("RMSPE_SET", "RMSPE_MSET", "RMSPE_SEQ")
   
   
   #non fitting
   non_fit_arr <- c(
      nrow(fold_events),
      nrow(events_anot),
      nrow(events_anot[events_anot$set_nf == 1,]),
      nrow(events_anot[events_anot$mset_nf == 1,]),
      nrow(events_anot[events_anot$seq_nf == 1,]),
      length(unique(events_anot$set_state_id)),
      length(unique(events_anot$mset_state_id)),
      length(unique(events_anot$seq_state_id))
   )
   names(non_fit_arr) <- c("TOT_EVT","TOT_EVT_OK",
                           "NF_SET", "NF_MSET","NF_SEQ", 
                           "STATES_SET", "STATES_MSET", "STATES_SEQ")
   non_fit_per_arr <- c(
      non_fit_arr[c("NF_SET")] / non_fit_arr[c("TOT_EVT_OK")] * 100,
      non_fit_arr[c("NF_MSET")] / non_fit_arr[c("TOT_EVT_OK")] * 100,
      non_fit_arr[c("NF_SEQ")] / non_fit_arr[c("TOT_EVT_OK")] * 100
   )
   names(non_fit_per_arr) <- c("NF_PERC_SET","NF_PERC_MSET","NF_PERC_SEQ")
   
   # Alexandre: precisa remover esse cálculo aqui, não faz nenhum sentido repetir esse valor
   # vide comentário na função original - isso era um mínimo entre média e mediana
   #perr_tot_arr <- c(mape_val[c("val_mape_pset_mean")],
   #                  mape_val[c("val_mape_pmset_mean")],
   #                  mape_val[c("val_mape_pseq_mean")])
   #
   #names(perr_tot_arr) <- c("perr_tot_set","perr_tot_mset","perr_tot_seq")
   
   # Substituindo número anterior pelo desvio padrão do erro
   error_sd <- c(
      sd(events_anot_filtered$remaining_stc-events_anot_filtered$predict_remain_set),
      sd(events_anot_filtered$remaining_stc-events_anot_filtered$predict_remain_mset),
      sd(events_anot_filtered$remaining_stc-events_anot_filtered$predict_remain_seq)
   )
   names(error_sd) <- c("error_sd_set","error_sd_mset","error_sd_seq")
   
   # CALCULATES FOR THE PREDICTION USING THE SOJOURN
   
   #MAPE(y_pred, y_true)
   mape_sj_val <- c(
      MAPE(events_anot_filtered$predict_remain_set_sj, events_anot_filtered$remaining_stc),
      MAPE(events_anot_filtered$predict_remain_mset_sj, events_anot_filtered$remaining_stc),
      MAPE(events_anot_filtered$predict_remain_seq_sj, events_anot_filtered$remaining_stc)
   )
   mape_sj_val <- mape_sj_val * 100
   names(mape_sj_val) <- c("MAPE_PERC_SET_SJ","MAPE_PERC_MSET_SJ","MAPE_PERC_SEQ_SJ")
   
   #RMSPE(y_pred, y_true)
   rmspe_sj_val <- c(
      RMSPE(events_anot_filtered$predict_remain_set_sj, events_anot_filtered$remaining_stc),
      RMSPE(events_anot_filtered$predict_remain_mset_sj, events_anot_filtered$remaining_stc),
      RMSPE(events_anot_filtered$predict_remain_seq_sj, events_anot_filtered$remaining_stc)
   )
   names(rmspe_sj_val) <- c("RMSPE_SET_SJ", "RMSPE_MSET_SJ", "RMSPE_SEQ_SJ")
   
   
   # filtro para eventos com fit
   events_anot_filtered_fitted_set <- events_anot_filtered[events_anot_filtered$set_state_id != NIL_STATE,]
   events_anot_filtered_fitted_mset <- events_anot_filtered[events_anot_filtered$mset_state_id != NIL_STATE,]
   events_anot_filtered_fitted_seq <- events_anot_filtered[events_anot_filtered$seq_state_id != NIL_STATE,]
   
   mape_val1 <- c(
      MAPE(events_anot_filtered_fitted_set$predict_remain_set, events_anot_filtered_fitted_set$remaining_stc),
      MAPE(events_anot_filtered_fitted_mset$predict_remain_mset, events_anot_filtered_fitted_mset$remaining_stc),
      MAPE(events_anot_filtered_fitted_seq$predict_remain_seq, events_anot_filtered_fitted_seq$remaining_stc)
   )
   names(mape_val1) <- c("MAPE_FIT_SET","MAPE_FIT_MSET","MAPE_FIT_SEQ")
   
   #RMSPE(y_pred, y_true)
   rmspe_val1 <- c(
      RMSPE(events_anot_filtered_fitted_set$predict_remain_set, events_anot_filtered_fitted_set$remaining_stc),
      RMSPE(events_anot_filtered_fitted_mset$predict_remain_mset, events_anot_filtered_fitted_mset$remaining_stc),
      RMSPE(events_anot_filtered_fitted_seq$predict_remain_seq, events_anot_filtered_fitted_seq$remaining_stc)
   )
   names(rmspe_val1) <- c("RMSPE_FIT_SET", "RMSPE_FIT_MSET", "RMSPE_FIT_SEQ")
   
   mapePerc <- mape_val * 100
   names(mapePerc) <- c("MAPE_PERC_SET","MAPE_PERC_MSET","MAPE_PERC_SEQ")
   
   if (type == "V") {
      exact_match <- c(length(which(events_anot$set_nf==0)),
                       length(which(events_anot$mset_nf==0)),
                       length(which(events_anot$seq_nf==0)))
   } else {
      exact_match <- rep(nrow(events_anot),3)
   }   
   
   names(exact_match) <- c("EXACT_MATCH_SET", "EXACT_MATCH_MSET", "EXACT_MATCH_SEQ")

   
      
   # appends to the file that contains all the stats so the error can be recalculated later
   events_anot <- cbind(type,fold,horiz,events_anot)
   write.table(events_anot, file=resultFile, row.names=FALSE, col.names = TRUE, append=TRUE, sep=";", dec=",")
   
   # returns only the summarized results for the given fold and horizon
   result <- c(fold=type, horizon=horiz, mape_val, non_fit_arr, non_fit_per_arr, error_sd, 
               mape_val1, rmspe_val, rmspe_val1, mapePerc, exact_match, mape_sj_val, rmspe_sj_val)
   
   return(result)
   
}


#' Fórmula 1 para o cálculo da predição, que leva em conta o sojourn atual e a média
#' de sojourn no estado. É a fórmula do código original.
#'
#' @param events_anot
#'
#' @return
#' @export
#'
#' @examples
calculate_prediction_with_sojourn <- function(aevents_anot) {
   
   events_anot <- aevents_anot
   
   events_anot$predict_remain_set_sj <-
      events_anot$set_state_remain_stc +
      events_anot$set_state_sj_stc -
      events_anot$sojourn_set_stc
   
   events_anot$predict_remain_mset_sj <-
      events_anot$mset_state_remain_stc +
      events_anot$mset_state_sj_stc -
      events_anot$sojourn_mset_stc
   
   events_anot$predict_remain_seq_sj <-
      events_anot$seq_state_remain_stc +
      events_anot$seq_state_sj_stc -
      events_anot$sojourn_seq_stc
   
   eval.parent(substitute(aevents_anot<-events_anot))
   
}

#' Fórmula 2, que leva em conta apenas a média do remaining.
#'
#' @param events_anot
#'
#' @return
#' @export
#'
#' @examples
calculate_prediction <- function(aevents_anot) {
   
   events_anot <- aevents_anot
   
   events_anot$predict_remain_set <- events_anot$set_state_remain_stc
   events_anot$predict_remain_mset <- events_anot$mset_state_remain_stc
   events_anot$predict_remain_seq <- events_anot$seq_state_remain_stc
   
   eval.parent(substitute(aevents_anot<-events_anot))
   
}

reshapeStateList <- function(ds) {
   n <- max(sapply(ds, length))
   rds <- lapply(ds, function(X) {
      c(as.character(X), rep("", times = n - length(X)))
   })
   out <- do.call(rbind, rds)
   return(out)
}

save_model <- function(model, startTime, filePath, filePrefix)
{
   
   # SET data
   if (length(model$full_set) > 0) {
      sfilen <- file.path(filePath,paste(filePrefix,"_MODEL_SET.csv",sep=""))
      r <- reshapeStateList(model$full_set)
      write.table(r, file=sfilen, row.names=TRUE, col.names = FALSE, sep=";", dec=",")
   }
   # SET Summary
   sfilen <- file.path(filePath,paste(filePrefix,"_MODEL_SET_SUMM.csv",sep=""))
   summary <- merge(model$summary_set, model$summary_sj_set, by = "set_state_id", suffixes = c("", "_sj"))
   write.table(summary, file=sfilen, row.names=FALSE, col.names = TRUE, sep=";", dec=",")

   # MSET data
   if (length(model$full_mset) > 0) {
      sfilen <- file.path(filePath,paste(filePrefix,"_MODEL_MSET.csv",sep=""))
      r <- reshapeStateList(model$full_mset)
      write.table(r, file=sfilen, row.names=TRUE, col.names = FALSE, sep=";", dec=",")
   }
   # MSET summary
   sfilen <- file.path(filePath,paste(filePrefix,"_MODEL_MSET_SUMM.csv",sep=""))
   summary <- merge(model$summary_mset, model$summary_sj_mset, by = "mset_state_id", suffixes = c("", "_sj"))
   write.table(summary, file=sfilen, row.names=FALSE, col.names = TRUE, sep=";", dec=",")

   # SEQ data
   if (length(model$full_seq) > 0) {
      sfilen <- file.path(filePath,paste(filePrefix,"_MODEL_SEQ.csv",sep=""))
      r <- reshapeStateList(model$full_seq)
      write.table(r, file=sfilen, row.names=TRUE, col.names = FALSE, sep=";", dec=",")
   }
   # SEQ Summary
   sfilen <- file.path(filePath,paste(filePrefix,"_MODEL_SEQ_SUMM.csv",sep=""))
   summary <- merge(model$summary_seq, model$summary_sj_seq, by = "seq_state_id", suffixes = c("", "_sj"))
   write.table(summary, file=sfilen, row.names=FALSE, col.names = TRUE, sep=";", dec=",")
   
   # General Parameters
   parameters <- c("SEL ATTRIBUTES", paste(unlist(model$sel_attributes),collapse=","))
   parameters <- rbind(parameters, c("HORIZON", model$horiz))
   parameters <- rbind(parameters, c("START TIME", format(startTime, "%Y-%m-%d %H:%M:%S")))
   parameters <- rbind(parameters, c("EXECUTION_DESCRIPTION", EXECUTION_DESCRIPTION))
   parameters <- rbind(parameters, c("REMOVE_LONGER_CASES", REMOVE_LONGER_CASES))
   parameters <- rbind(parameters, c("SEARCH_SIMILAR", SEARCH_SIMILAR))
   parameters <- rbind(parameters, c("SIM_METHOD_SET", SIM_METHOD_SET))
   parameters <- rbind(parameters, c("SIM_METHOD_MSET", SIM_METHOD_MSET))
   parameters <- rbind(parameters, c("SIM_METHOD_SEQ", SIM_METHOD_SEQ))

   sfilen <- file.path(filePath,paste(filePrefix,"_MODEL_PARAMS.csv",sep=""))
   write.table(parameters, file=sfilen, row.names=FALSE, col.names = FALSE, sep=";", dec=",")
   
}

