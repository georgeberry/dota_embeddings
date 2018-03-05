## Embed Dota 2 heroes in two dimensions

The process is simple here: embed heroes just like words, but pick a hidden dimension of size 2. We could pick a larger number of latent dimensions for other tasks, but my goal was to quickly visualize the heroes in 2 dimensions. An alternative strategy would be to embed heroes in > 2 dimensions and then use a t-SNE for dimensionality reduction.

Also for simplicity I embed from the Radiant perspective, so if you embed dire heroes you might get slightly different results. I ignore order pick order effects, which could also add interesting complexity.

![picture](https://github.com/georgeberry/dota_embeddings/blob/master/data/p1.png)

### Packages you'll need

Python 3

- pytorch
- requests
- numpy
- pandas

R

- ggplot / tidyverse
- ggrepel

### How to run

1. Make sure your paths are right. Either naviagte to the `dota_embeddings` folder or change paths to be absolute.
2. Run `crawl.py` for awhile
3. Run `embed.py`
4. Open up R and visualize using `visualize.R`
