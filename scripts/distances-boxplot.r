library(ggplot2)
library(dplyr)
library(scales)    
df <- read.csv("/media/cezarsas/Data/PyCharmProjects/ComponentSemantics/componentSemantics/distances.csv", stringsAsFactors = FALSE)
df$level[df$level=="package"] <- "BERT-Pack"
df$level[df$level=="document"] <- "BERT-Doc"
df$level[df$level=="TFIDF"] <- "TF-IDF"

df$level = factor(df$level, levels=c("BERT-Pack", "BERT-Doc", "TF-IDF", "fastText", "code2vec"))
df$algorithm = factor(df$algorithm, levels=c("leiden", "infomap")) 

ggplot(df, aes(x = level, y = dist, color = algorithm)) +
  geom_violin(fill = "grey80") +
  labs(x = element_blank(), y = "Distance") +
  theme_linedraw() +
  theme(legend.title = element_blank(),
        legend.position = c(0.15, 0.9),
        legend.text = element_text(size=18),
        legend.background = element_rect(fill=alpha('white', 0)),
        panel.grid.major.y = element_line(color = "grey80"),
        panel.grid.major.x = element_line(color = "grey80"),
        panel.grid.minor.x = element_blank(),
        panel.grid.minor.y = element_blank(),
        axis.text=element_text(size=18),
        axis.title=element_text(size=18))

ggsave("/media/cezarsas/Data/PyCharmProjects/ComponentSemantics/componentSemantics/distances.pdf", width = 8, height = 6)