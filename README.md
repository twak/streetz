# street statistics

instructions:

* Create a folder in root called `npz` your data. Number of files is limited by number colors.
* Run `stats.py`. Runtime is about 2mins per graph. An output figure is created using 
* Edit `COLORS` in `utils.py` to adjust coloring
* Edit the `metric_fns` array in `stats.py` at around line 229 to control which statistics are run. The graphs function (transport ration, betweenness, & pagerank) are the slow ones.
* Latex formatted table is written 
