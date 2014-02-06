XLStoTXT <- function(path='~/STAT-222/Raw Datasets/SCAD 3.0/SCAD 3.0 1990-2011.xls'){
  require('gdata')
  SCAD <- read.xls(path, method='tab')
  SCAD[SCAD==""] <- NA
  txtPath <- gsub('.xls', '.txt', path)
  write.table(SCAD, txtPath, sep='\t')
}