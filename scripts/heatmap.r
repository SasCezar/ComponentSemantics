library(ggplot2)
library(dplyr)

d <- read.csv('/home/sasce/PycharmProjects/ComponentSemantics/scripts/heatmap_class_sim_3rd.csv')

ggplot(d, aes(SRC, TRG, fill = Value)) +
  geom_tile() +
  theme(axis.text.x = element_text(angle = 45, hjust=1))