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
build_prediction_optimized <- function(aevents, ats)
{
   events <- aevents  # needed to allow update the original list in the caller scope
   
   events$seq_state_id <- 0
   events$set_state_id <- 0
   events$mset_state_id <- 0
   events$sojourn_set_stc <- 0
   events$sojourn_mset_stc <- 0
   events$sojourn_seq_stc <- 0
   # initializes the flags to indicate if the event is going for a non-fitting path
   events$set_nf <- 0
   events$mset_nf <- 0
   events$seq_nf <- 0
   
   # one function for each situation to improve performance, avoiding the validation for every event
   if ( SEARCH_SIMILAR ) {
      if ( ats$horiz==1 )
         pred <- build_prediction_horizon_one_similar(events,ats)
      else 
         pred <- build_prediction_similar(events,ats)
   } else { 
      if ( ats$horiz==1 )
         pred <- build_prediction_horiz_one_exact(events,ats)
      else 
         pred <- build_prediction_exact(events,ats)
   }
   eval.parent(substitute(aevents<-events))
   
   return(pred)
}


# ---- Horizon One, Exact Search ------   


#' Version of build prediction specific for horizon ONE and exact match search
#'
#' @param aevents 
#' @param ats 
#'
#' @return
#' @export
#'
#' @examples
build_prediction_horiz_one_exact <- function(aevents, ats) 
{
   
   generate_log("Starting build_prediction for horizon one, exact search",2)
   
   events <- aevents  # needed to allow update the original list in the caller scope
   
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
   
   for(i in 1:nrow(traces))
   {
      # busca eventos do caso / search events of the given trace
      traceEvents <- events[events$number == traces$number[i],]
      
      # label utilizado na funcao de eventos (campos do modelo)
      activity_fields <- do.call(paste, as.data.frame(traceEvents[,ats$sel_attributes]))
      
      # listas com os modelos de abstracao usados para representacao dos eventos
      seq_list <- list()
      set_list <- list()
      mset_list <- list()
      for (j in 1:nrow(traceEvents))
      {
         # calculate the horizon
         horiz_index <- max(j - ats$horiz + 1, 1)
         
         # generates state for SEQUENCE abstration
         inc_seq <- as.character(activity_fields[horiz_index:j])
         seq_list[[j]]  <- toString(inc_seq)

         # searches for the exact match
         stateId <- match(seq_list[[j]], seq_state_list, NIL_STATE)
         
         traceEvents[j,c("set_state_id","mset_state_id","seq_state_id")] <- stateId

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

   } # end loop over all the traces
   
   # for the exact search, all events with state -1 are non-fitting
   if (nrow(events[events$set_state_id == NIL_STATE,]) > 0)
      events[events$set_state_id == NIL_STATE,]$set_nf <- 1
   if (nrow(events[events$mset_state_id == NIL_STATE,]) > 0)
      events[events$mset_state_id == NIL_STATE,]$mset_nf <- 1
   if (nrow(events[events$seq_state_id == NIL_STATE,]) > 0)
      events[events$seq_state_id == NIL_STATE,]$seq_nf <- 1
   
   # retorna resultado - precisa atualizar a lista de eventos original, para os cálculos
   eval.parent(substitute(aevents<-events))
   
   mta_model <- list(traces_states=traces_states,
                     horiz=ats$horiz,
                     sel_attributes=ats$sel_attributes
   )
   
   generate_log("Ended build_prediction",2)
   
   return(mta_model)
}

# ---- Horizon > 1, Exact Search ------   

#' Version of build prediction specific for horizon > 1 and exact match search
#'
#' @param aevents 
#' @param ats 
#'
#' @return
#' @export
#'
#' @examples
build_prediction_exact <- function(aevents, ats) 
{
   
   generate_log("Starting build_prediction for horizon > 1, exact search",2)

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
   
   for(i in 1:nrow(traces))
   {
      # busca eventos do caso / search events of the given trace
      traceEvents <- events[events$number == traces$number[i],]
      
      # label utilizado na funcao de eventos (campos do modelo)
      activity_fields <- do.call(paste, as.data.frame(traceEvents[,ats$sel_attributes]))
      
      # listas com os modelos de abstracao usados para representacao dos eventos
      seq_list <- list()
      set_list <- list()
      mset_list <- list()
      for (j in 1:nrow(traceEvents))
      {
         # calculate the horizon
         horiz_index <- max(j - ats$horiz + 1, 1)
         
         # generates state for SEQUENCE abstration
         inc_seq <- as.character(activity_fields[horiz_index:j])

         seq_list[[j]]  <- toString(inc_seq)

         # searches for the exact match
         stateId <- match(seq_list[[j]], seq_state_list, NIL_STATE)
         traceEvents[j,"seq_state_id"] <- stateId
            
            # generates state for SET
         inc_set <- as.set(inc_seq)
         set_list[[j]]  <- toString(inc_set)
         
         # executes the exact search
         stateId <- match(set_list[[j]], set_state_list, NIL_STATE)
         traceEvents[j,"set_state_id"] <- stateId
         
            # multi-set (requires package sets)
         inc_gset <- as.gset(inc_seq)
         inc_gset_str <- toString(rbind(gset_support(inc_gset), gset_memberships(inc_gset)))
         mset_list[[j]]  <- inc_gset_str            
         
         # executes the exact search
         stateId <- match(mset_list[[j]], mset_state_list, NIL_STATE)
         traceEvents[j,"mset_state_id"] <- stateId

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
      
   } # end loop over all the traces

   # for the exact search, all events with state -1 are non-fitting
   if (nrow(events[events$set_state_id == NIL_STATE,]) > 0)
      events[events$set_state_id == NIL_STATE,]$set_nf <- 1
   if (nrow(events[events$mset_state_id == NIL_STATE,]) > 0)
      events[events$mset_state_id == NIL_STATE,]$mset_nf <- 1
   if (nrow(events[events$seq_state_id == NIL_STATE,]) > 0)
      events[events$seq_state_id == NIL_STATE,]$seq_nf <- 1
   

   # retorna resultado - precisa atualizar a lista de eventos original, para os cálculos
   eval.parent(substitute(aevents<-events))
   
   mta_model <- list(traces_states=traces_states,
                     horiz=ats$horiz,
                     sel_attributes=ats$sel_attributes
   )
   
   generate_log("Ended build_prediction",2)
   
   return(mta_model)
}
   

# ---- Horizon One, Similar Search ------   

#' Version of build prediction specific for horizon one and similar match search
#'
#' @param aevents 
#' @param ats 
#'
#' @return
#' @export
#'
#' @examples
build_prediction_horizon_one_similar <- function(aevents, ats) 
{
   
   generate_log("Starting build_prediction for horizon ONE, similar search",2)
   
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
   full_sequence <- ats$full_sequence

   for(i in 1:nrow(traces))
   {
      # busca eventos do caso / search events of the given trace
      traceEvents <- events[events$number == traces$number[i],]
      
      # label utilizado na funcao de eventos (campos do modelo)
      activity_fields <- do.call(paste, as.data.frame(traceEvents[,ats$sel_attributes]))
      
      # listas com os modelos de abstracao usados para representacao dos eventos
      seq_list <- list()
      all_in_non_fit_path <- FALSE
      for (j in 1:nrow(traceEvents))
      {
         # calculate the horizon
         horiz_index <- max(j - ats$horiz + 1, 1)
         
         # generates state for SEQUENCE abstration
         inc_seq <- as.character(activity_fields[horiz_index:j])
         # there os no need to rebuild the search string for horizon one
         # since the previous steps are not represented
         seq_list[[j]]  <- toString(inc_seq)
         
         # searches for the exact match - this is necessary only once, for the SEQ (see explanation below)
         stateId <- match(seq_list[[j]], seq_state_list, NIL_STATE)
         all_in_non_fit_path <- ( stateId == NIL_STATE )
         if(all_in_non_fit_path) {
            
            traceEvents[j,c("set_nf","mset_nf","seq_nf")] <- 1
            
            # steps below are needed to allow for use of distinct search methods
            # but the list is the SEQ, which is the same for all. For that reason the
            # searched string must also be the same as for SEQ. This is not a problem, 
            # since for horizon one there are no differences between SEQ, SET and MSET
            
            # search for similar using the method for SEQ
            #stateId <- which.max(stringsim(a=seq_list[[j]], b=unlist(seq_state_list), method=SIM_METHOD_SEQ, useBytes=FALSE))
            if (SIM_METHOD_SEQ=="jaccard") {
               stateId <- which.max(jaccardSimList(full_sequence[[j]],full_sequence))
            } else {
               stateId <- which.max(stringsim(a=seq_list[[j]], b=unlist(seq_state_list), method=SIM_METHOD_SEQ, useBytes=FALSE))
            }
            traceEvents[j,"seq_state_id"] <- stateId

            # search for similar using the method for SET
            #stateId <- which.max(stringsim(a=seq_list[[j]], b=unlist(seq_state_list), method=SIM_METHOD_SET, useBytes=FALSE))
            if (SIM_METHOD_SET=="jaccard") {
               stateId <- which.max(jaccardSimList(full_sequence[[j]],full_sequence))
            } else {
               stateId <- which.max(stringsim(a=seq_list[[j]], b=unlist(seq_state_list), method=SIM_METHOD_SET, useBytes=FALSE))            
            }
            traceEvents[j,"set_state_id"] <- stateId
            
            # search for similar using the method for MSET
            #stateId <- which.max(stringsim(a=seq_list[[j]], b=unlist(seq_state_list), method=SIM_METHOD_MSET, useBytes=FALSE))
            if (SIM_METHOD_MSET=="jaccard") {
               stateId <- which.max(jaccardSimList(full_sequence[[j]],full_sequence))
            } else {
               stateId <- which.max(stringsim(a=seq_list[[j]], b=unlist(seq_state_list), method=SIM_METHOD_MSET, useBytes=FALSE))
            }
            traceEvents[j,"mset_state_id"] <- stateId
            
         } else {
            traceEvents[j,c("set_state_id","mset_state_id","seq_state_id")] <- stateId
         }

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
                     horiz=ats$horiz,
                     sel_attributes=ats$sel_attributes
   )
   
   generate_log("Ended build_prediction",2)
   
   return(mta_model)
}

# ---- Horizon > 1, Similar Search ------   

#' Version of build prediction specific for horizon > 1 and similar match search
#'
#' @param aevents 
#' @param ats 
#'
#' @return
#' @export
#'
#' @examples
build_prediction_similar <- function(aevents, ats) 
{
   
   generate_log("Starting build_prediction for horizon > 1, similar search",2)
   
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
   full_sequence <- ats$full_sequence
   if ( ats$horiz==1 ) {
      # for horizon one just the sequence list exists
      set_state_list <- ats$seq_state_list
      mset_state_list <- ats$seq_state_list
      full_set <- ats$full_sequence
      full_mset <- ats$full_sequence
   } else {
      set_state_list <- ats$set_state_list
      mset_state_list <- ats$mset_state_list
      full_set <- ats$full_set
      full_mset <- ats$full_mset
   }
   
   for(i in 1:nrow(traces))
   {
      # busca eventos do caso / search events of the given trace
      traceEvents <- events[events$number == traces$number[i],]
      
      # label utilizado na funcao de eventos (campos do modelo)
      activity_fields <- do.call(paste, as.data.frame(traceEvents[,ats$sel_attributes]))
      
      # listas com os modelos de abstracao usados para representacao dos eventos
      seq_list <- list()
      set_list <- list()
      mset_list <- list()
      seq_in_non_fit_path <- set_in_non_fit_path <- mset_in_non_fit_path <- FALSE
      similar_seq_id <- similar_set_id <- similar_mset_id <- NIL_STATE
      for (j in 1:nrow(traceEvents))
      {
         # calculate the horizon
         horiz_index <- max(j - ats$horiz + 1, 1)
         
         # generates state for SEQUENCE abstration
         inc_seq <- as.character(activity_fields[horiz_index:j])
         # to handle the non-fitting path 
         inc_seq_nf <- as.character((activity_fields[j]))
         
         if (seq_in_non_fit_path) {
            new_value <- full_sequence[[similar_seq_id]]
            new_value <- append(new_value,inc_seq_nf)
            seq_list[[j]]  <- toString(new_value)
         } else {
            seq_list[[j]]  <- toString(inc_seq)
         }
         
         # searches for the exact match
         stateId <- match(seq_list[[j]], seq_state_list, NIL_STATE)
         seq_in_non_fit_path <- ( stateId == NIL_STATE )
         if(seq_in_non_fit_path) {
            traceEvents[j,"seq_nf"] <- 1
            #stateId <- which.max(stringsim(a=seq_list[[j]], b=unlist(seq_state_list), method=SIM_METHOD_SEQ, useBytes=FALSE))
            if (SIM_METHOD_SEQ=="jaccard") {
               stateId <- which.max(jaccardSimList(full_sequence[[j]],full_sequence))
            } else {
               stateId <- which.max(stringsim(a=seq_list[[j]], b=unlist(seq_state_list), method=SIM_METHOD_SEQ, useBytes=FALSE))
            }
            similar_seq_id <- stateId
         }
         traceEvents[j,"seq_state_id"] <- stateId
         
         # generates state for SET
         if (set_in_non_fit_path) {
            inc_set <- full_set[[similar_set_id]]
            inc_set <- append(inc_set,inc_seq_nf)
         } else {
            inc_set <- as.set(inc_seq)
         }
         set_list[[j]]  <- toString(inc_set)
         
         # executes the exact search
         stateId <- match(set_list[[j]], set_state_list, NIL_STATE)
         set_in_non_fit_path <- ( stateId == NIL_STATE )
         if (set_in_non_fit_path) {
            # handling the non-fitting - using the DL distance to get the most similar entry
            traceEvents[j,"set_nf"] <- 1
            #stateId <- which.max(stringsim(a=set_list[[j]], b=unlist(set_state_list), method=SIM_METHOD_SET, useBytes=FALSE))
            if (SIM_METHOD_SET=="jaccard") {
               stateId <- which.max(jaccardSimList(full_set[[j]],full_set))
            } else {
               stateId <- which.max(stringsim(a=set_list[[j]], b=unlist(set_state_list), method=SIM_METHOD_SET, useBytes=FALSE))
            }
            similar_set_id <- stateId
         }
         traceEvents[j,"set_state_id"] <- stateId
         
         # multi-set (requires package sets)
         if (mset_in_non_fit_path) {
            inc_gset <- full_mset[[similar_mset_id]]
            inc_gset <- append(inc_gset,inc_seq_nf)
            inc_gset <- as.gset(inc_gset)
         } else {
            inc_gset <- as.gset(inc_seq)
         }
         
         inc_gset_str <- toString(rbind(gset_support(inc_gset), gset_memberships(inc_gset)))
         mset_list[[j]]  <- inc_gset_str            
         
         # executes the exact search
         stateId <- match(mset_list[[j]], mset_state_list, NIL_STATE)
         mset_in_non_fit_path <- ( stateId == NIL_STATE )
         if(mset_in_non_fit_path) {
            traceEvents[j,"mset_nf"] <- 1
            #stateId <- which.max(stringsim(a=mset_list[[j]], b=unlist(mset_state_list), method=SIM_METHOD_MSET, useBytes=FALSE))
            if (SIM_METHOD_MSET=="jaccard") {
               stateId <- which.max(jaccardSimList(full_mset[[j]],full_mset))
            } else {
               stateId <- which.max(stringsim(a=mset_list[[j]], b=unlist(mset_state_list), method=SIM_METHOD_MSET, useBytes=FALSE))
            }
            similar_mset_id <- stateId
         }
         traceEvents[j,"mset_state_id"] <- stateId

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
                     horiz=ats$horiz,
                     sel_attributes=ats$sel_attributes
   )
   
   generate_log("Ended build_prediction",2)
   
   return(mta_model)
}

#' Calculates the Jaccard similarity index between the unique elements of the two given vectors.
#' Formula is Jaccard = intersection(v1,v2) / union(v1,v2)
#'
#' @param v1 first vector - usually the one you are looking for
#' @param v2 second vector - usually your base list
#'
#' @return double value in the range [0-1] being: 
#'       [0]: indicates that there is no common elements, thus there is no similarity
#'       [1]: indicates that all elements of one vector are present
#' @export
#'
#' @examples
jaccardSimil <- function(v1, v2) {
   if ( length(v1) > 0 & length(v2 > 0 ) ) {
      i <- length(intersect(v1,v2))
      if ( i > 0 ) {
         return ( i / length(union(v1, v2)) ) 
      } 
   }
   return(0)
}


#' Wrapper for the jaccardSimil() function, calling it for each one of the elements
#' of the second vector. The first vector will be the also in this position in the inner call.
#' The vList is assumed to be a list of lists, thus the elements are iterated as vList[[i]]
#'
#' @param v 
#' @param vList 
#'
#' @return
#' @export
#'
#' @examples
jaccardSimList <- function(v, vList) {
   jList <- NULL
   for(i in 1:length(vList)) {
      jList <- c(jList,jaccardSimil(v,vList[[i]]))
   }
   return(jList)
}

