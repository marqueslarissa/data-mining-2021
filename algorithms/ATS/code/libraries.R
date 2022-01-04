
# # --- load packages ----
library(corrplot)
library(plyr) # before Hmisc, dplyr
library(Hmisc) # before dplyr, fields
library(tidyr) # before sets
library(sets) # before dplyr, data.table
library(dplyr) # pbefore data.table
library(data.table)
library(doParallel)
library(fields)
library(lsr)
library(MLmetrics)
library(nortest)
library(readr)
library(writexl)
library(stringdist)

# # --- install packages ----
# needed only in the first time
# 
# install.packages("corrplot")
# install.packages("plyr")
# install.packages("Hmisc")
# install.packages("tidyr")
# install.packages("sets")
# install.packages("dplyr")
# install.packages("data.table")
# install.packages("doParallel")
# install.packages("fields")
# install.packages("lsr")
# install.packages("MLmetrics")
# install.packages("nortest")
# install.packages("readr")
# install.packages("writexl")
# install.packages("stringdist")

# for readr version problem, need to upgrade hms but not to the source version
# as it also requires update other libraries from source
#install.packages("hms",type="win.binary")



