XLStoTXT <- function(path){
  require('gdata')
  data <- read.xls(path, method='tab')
  data[data==""] <- NA
  txtPath <- gsub('.xls', '.txt', path)
  write.table(data, txtPath, sep='\t')
}