# street statistics

instructions:

* Create a folder in root called `npz` your data. Number of files is limited by number colors.
* Run `stats.py`. Runtime is about 2mins per graph. The output figure is created using `plt.show()`.
* Edit `COLORS` in `utils.py` to adjust coloring
* Edit the `metric_fns` array in `stats.py` at around line `229` to control which statistics are run. The graphs function (transport ration, betweenness, & pagerank) are the slow ones.
* Latex formatted table is written to `table.tex` in the root folder.
* `SAMPLES` in `stats_graph.py` controls the number of shortest paths calculated for transport ratio and betweenness

implementation notes:
* a segment is a maximum sequence of adjacent edges between vertices which have a valency != 2 (i.e., segments can bend 90 degrees, as long as they don't go through junctions).
* Segment circuity measures the ratio of the euclidean distance between the start and end of a segment and its length. e.g., how curved the segments are.
* Edge/vertex/segment/block density normalises by the area of land (and not water) in the map. 
  * Note some edges (bridges!) will break this assumption
* A block is every enclosed region in the graph without any internal edges.
We don't process holes, so a loop inside a block counts as an overlapping block.
  * I assume the graph is planar and edges don't cross
  * The aspect ratio of a block is found by finding the smallest rectangle which covers all vertices. The aspect ratio of the rectangle's height/width is reported.
* Rectangularness is the ratio of the true block area to the smallest rectangle. A value of 1 means all blocks are rectangular.
* Transport ratio is calculated between two vertices and is the ratio of the shortest path length to euclidean distance. It is slow to compute the true value, so we sample random vertex pairs. Values seem to converge at around 300 iterations…but smoother graphs with higher values.
  * Random walks are between two randomly selected nodes. Evaluated stochastically
* Betweenness-centrality measures the number of times that each vertex is visited on shortest paths between all combinations of start and end nodes. Again for speed purposes, this is sampled stochastically.
  https://networkx.org/documentation/stable/reference/algorithms/centrality.html
  * Maximum Betweenness Centrality measures the share of shortest paths which pass through the network's most important node. Again, stochastic.
* Pagerank
  * intended to measure integration of streets
  * google/lary page's algorithm on topology only (i.e., ignoring street lengths)
  * pagerank-on-edges uses the street edges as nodes (connected to other edges at their start and end), so we can initialize them with a score proportional to their length

in progress:
  * normalise "number of ###' as a density using land-area fraction
  * Graphs for the above where it makes sense
  
references:
 * https://arxiv.org/pdf/1705.02198.pdf
 * https://appliednetsci.springeropen.com/track/pdf/10.1007/s41109-019-0189-1.pdf
 * transport ratio https://arxiv.org/ftp/arxiv/papers/1708/1708.00836.pdf
