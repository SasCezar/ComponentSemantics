library(ggplot2)
library(dplyr)
require(gridExtra)

df <- read.csv("/media/cezarsas/Data/PyCharmProjects/ComponentSemantics/componentSemantics/num_comm.csv", stringsAsFactors = FALSE)
df$level[df$level=="package"] <- "BERT-Pack"
df$level[df$level=="document"] <- "BERT-Doc"
df$level = factor(df$level, levels=c("BERT-Pack", "BERT-Doc", "TF-IDF", "fastText"))
df$algorithm = factor(df$algorithm, levels=c("leiden", "infomap")) 

num_comm <- ggplot(df, aes(x = algorithm, y = num, color = algorithm)) +
  geom_violin(outlier.shape = NA,fill="grey80") +
  coord_cartesian(ylim = quantile(df$num, c(0.1, 0.99))) +
  labs(x = element_blank(), y = "Number Communities") +
  theme(legend.title = element_blank()) +
  theme_linedraw() +
  theme(legend.position = "none", 
        panel.grid.major.y = element_line(color = "grey80"),
        panel.grid.major.x = element_line(color = "grey80"),
        panel.grid.minor.x = element_blank(),
        panel.grid.minor.y = element_blank(),
        axis.text=element_text(size=12),
        axis.title=element_text(size=14))

df <- read.csv("/media/cezarsas/Data/PyCharmProjects/ComponentSemantics/componentSemantics/comm_sizes.csv", stringsAsFactors = FALSE)
df$level[df$level=="package"] <- "BERT-Pack"
df$level[df$level=="document"] <- "BERT-Doc"
df$level = factor(df$level, levels=c("BERT-Pack", "BERT-Doc", "TF-IDF", "fastText"))
df$algorithm = factor(df$algorithm, levels=c("leiden", "infomap")) 

comm_sizes <- ggplot(df, aes(x = algorithm, y = size, color = algorithm)) +
  geom_violin(outlier.shape = NA,fill="grey80") +
  coord_cartesian(ylim = quantile(df$num, c(0.1, 0.99))) +
  labs(x = element_blank(), y = "Communities Size") +
  theme(legend.title = element_blank()) +
  theme_linedraw() +
  theme(legend.position = "none", 
        panel.grid.major.y = element_line(color = "grey80"),
        panel.grid.major.x = element_line(color = "grey80"),
        panel.grid.minor.x = element_blank(),
        panel.grid.minor.y = element_blank(),
        axis.text=element_text(size=12),
        axis.title=element_text(size=14))

ggsave("/media/cezarsas/Data/PyCharmProjects/ComponentSemantics/componentSemantics/num_comm_sizes.pdf", arrangeGrob(num_comm, comm_sizes, ncol=2), width = 6, height = 5)