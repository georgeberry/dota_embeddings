library(tidyverse)
library(ggrepel)

df = read_csv('data/viz.csv') %>%
    select(-X1) %>%
    mutate(Carry=factor(Carry))


p1 = df %>%
    ggplot(aes(x=x, y=y, label=name, color=Carry)) +
    geom_point() +
    geom_text_repel(size=3) +
    theme_bw() +
    theme(
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        legend.position = c(0.07, 0.8),
        axis.ticks.x=element_blank(),
        axis.ticks.y=element_blank(),
        axis.title.x=element_blank(),
        axis.title.y=element_blank(),
        axis.text.x=element_blank(),
        axis.text.y=element_blank()
    ) +
    annotate(
        "text",
        x = -2,
        y = 2.2,
        label = "Dota 2 heroes represented in two dimensions\nMatches > 4.5k mmr"
    )

ggsave('data/p1.png', p1, height=6, width=10)
