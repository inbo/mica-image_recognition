### ----------------------------------------------------------------------------
### Importing images from Agouti into folders
### Emma Cartuyvels Sander Devisscher
### ----------------------------------------------------------------------------

library(readr)
library(dplyr)

#' Download images from Agouti to local disk
#' 
#' Data is downloaded to the `../data/raw` folder, with a subfolder for each deployment. The folder
#' name is the `deployment_id`.
#' 
#' @param sequence_ids "all" or list of sequence_id. The identifiers of the sequences to download.
#' @param deployment_ids "all" or list of deployment_id. The identifiers of the deployments to download.
#' @param n "all" or integer. Limit the number of downloads to `n`, useful for testing purposes.
#' 
#' @examples
#' download_images(sequence_ids = c("f9ad783b-f310-4560-86c0-86ceff691660", 
#'                                  "54a68c1c-a3b7-4bc1-9afa-8a2d270915a7"))
#' download_images(deployment_ids = "05c05f7b-6c38-4a06-a3b0-145de226c8ad", 
#'                 n = 20)
download_images <- function(sequence_ids = "all", deployment_ids = "all", n = "all"){

  assets <- read_csv(file = "../data/raw/multimedia.csv")

  assets$url <- as.character(assets$file_path)
  assets$originalFilename <- as.character(assets$file_name)

  data <- assets

  if(sequence_ids %in% c("all")){
    data <- data
  }else{
    data <- data %>%
      dplyr::filter(sequence_id %in% sequence_ids)
  }

  if(deployment_ids %in% c("all")){
    data <- data
  }else{
    data <- data %>%
      dplyr::filter(deployment_id %in% deployment_ids)
  }

  if(n == "all"){
    n <-  nrow(data)
  }else{
    if(!is.integer(n)){
      n  <-  as.integer(n)
      if(is.na(n)){
        stop("n - vallue is not usable, provide a integer or all")
      }
    }
  }

  print(nrow(data))

  for (i in 1:n){
    ifelse(dir.exists(paste0("../data/raw/",
                             data$deployment_id[i])),
           FALSE,
           dir.create(paste0("../data/raw/",
                             data$deployment_id[i])))
    download.file(data$url[i],
                  destfile = paste0("../data/raw/",
                                    data$deployment_id[i],
                                    "/",
                                    data$originalFilename[i]),
                  method = "curl")
  }
}
