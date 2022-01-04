# This global constant must be defined by the main project function
# It defines the log level that should be generated.
# The lower the number the less number of log output is generated.
LOG_LEVEL <- 2
LOG_FILE <- NULL
LOG_FILE_OPEN <- FALSE

set_log_file <- function(filename)
{
   eval.parent(substitute(LOG_FILE <- file(filename, open = "w")))
   eval.parent(substitute(LOG_FILE_OPEN <- TRUE))
}

close_log_file <- function()
{
   close(LOG_FILE)
   eval.parent(substitute(LOG_FILE <- NULL))
   eval.parent(substitute(LOG_FILE_OPEN <- FALSE))
}

#' Generates log
#'
#' @param texto
#' @param nivel
#'
#' @return
#' @export
#'
#' @examples
generate_log <- function(texto,nivel=1) {
   if (nivel <= LOG_LEVEL ) {
       s <- paste(Sys.time(), ":", texto)
      cat(s, sep="\n")
      if (LOG_FILE_OPEN) cat(s, file = LOG_FILE, append=TRUE, sep="\n")
   }
}





