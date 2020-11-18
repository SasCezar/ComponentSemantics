library(ggplot2)
library(dplyr)

get_proj_name <- function(x){
  return(basename(x))
}

projects <- list.dirs('/media/cezarsas/Data/PyCharmProjects/ComponentSemantics/data/arcanOutput', recursive=FALSE)
projects <- lapply(projects, get_proj_name)
methods <- list("infomap", "leiden")
embeddings <- list("package", "document", "TFIDF", "fastText")


for (project in projects){
  for (method in methods){
    for (embedding in embeddings){
      skip_to_next <- FALSE
      tryCatch(df <- read.csv(sprintf("/media/cezarsas/Data/PyCharmProjects/ComponentSemantics/data_out/plots/raw_data/TSNE_%s_%s_%s.csv", project, method, embedding)), error = function(e) { skip_to_next <<- TRUE})
      if(skip_to_next) { next } 
      
      df <- df %>% mutate(y = as.character(y))
      ggplot(df, aes(C1, C2,label = y, color = y)) + 
        # scale_shape_manual(values=0:12) +
        geom_point(size = 10, alpha = .5, shape = 20) + 
        geom_text(size = 5, color = "black", check_overlap = T) + 
        theme_linedraw() +
        theme(legend.position = "none", 
              panel.grid.major.y = element_line(color = "grey80"),
              panel.grid.major.x = element_line(color = "grey80"),
              panel.grid.minor.x = element_blank(),
              panel.grid.minor.y = element_blank(),
              axis.text = element_text(size = 14)) +
        # scale_color_grey() +
        labs(x = "Dimension 1", y = "Dimension 2") +
        labs(size = 14)
      
      out <- sprintf("/media/cezarsas/Data/PyCharmProjects/ComponentSemantics/data/plots/paper/TSNE_%s_%s_%s.pdf", project, method, embedding)
      ggsave(out, width = 8, height = 6)
    }
  }
}

