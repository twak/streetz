source for evaluation from the publication [Large-Scale Auto-Regressive Modeling Of Street Networks](https://arxiv.org/pdf/2209.00281.pdf).

# teaser figure

the street geometry was generated in CityEngine. I exported DWG from python, and imported into CE with a light clean-up.

* Blender file
* CityEngine Rules

# street statistics

instructions:

* create a folder in root called `npz` for your data. Number of files is limited by number colors.
* check the `scale_to_meters` parameter in `stats.py` will convert the npz data to meters (e.g, it should be the tile size)
* run `stats.py`. Runtime is about 2mins per graph. The output figure is created using `plt.show()`.
* edit `COLORS` in `utils.py` to adjust coloring
* edit the `metric_fns` array in `stats.py` at around line `229` to control which statistics are run. The graph metrics (transport ration, betweenness, & pagerank) are the slow ones.
* latex formatted table is written to `table.tex` in the root folder.
* `SAMPLES` in `stats_graph.py` controls the number of shortest paths calculated for transport ratio and betweenness

implementation notes:
* a segment is a maximum sequence of adjacent edges between vertices which have a valency != 2 (i.e., segments can bend 90 degrees, as long as they don't go through junctions).
* Segment circuity measures the ratio of the euclidean distance between the start and end of a segment and its length. e.g., how curved the segments are.
* Edge/vertex/segment/block density normalises by the area of land (and not water) in the map. 
  * Note some edges (bridges!) will break this assumption
* A block is every enclosed region in the graph without any internal edges.
  * We don't process holes, so a loop inside another block counts as two overlapping blocks.
  * I assume the graph is planar and edges don't cross
  * The aspect ratio of a block is found by finding the smallest rectangle which covers all vertices. The aspect ratio of the rectangle's height/width is reported.
* rectangularness is the ratio of the true block area to the smallest rectangle. A value of 1 means all blocks are rectangular.
* transport ratio is calculated between two vertices and is the ratio of the shortest path length to euclidean distance. It is slow to compute the true value, so we sample random vertex pairs. Values seem to converge at around 300 iterations…but smoother graphs with higher values. Render is currently poor at this number of samples: need to increase for nice maps.
* random walks are between two randomly selected nodes. Evaluated stochastically
* betweenness-centrality measures the number of times that each vertex is visited on shortest paths between all combinations of start and end nodes. Again for speed purposes, this is sampled stochastically.
  https://networkx.org/documentation/stable/reference/algorithms/centrality.html
  * Maximum Betweenness Centrality measures the share of shortest paths which pass through the network's most important node. Again, stochastic.
* pagerank
  * intended to measure integration of streets
  * google/lary page's algorithm. Using a higher k of 0.95. Topology only (i.e., ignoring street lengths)...so
  * ...pagerank-on-edges uses the street edges as nodes (connected to other edges at their start and end), so we can initialize them with a p proportional to their length. Looking at the graph, this just seems to prefer long edges (and penalized curves). Uses pagerank k of 0.95 to simulate longer "walks". Good at highlighting well connected regions.
  * The density values normalise by the land area.
  
references:
 * https://arxiv.org/pdf/1705.02198.pdf
 * https://appliednetsci.springeropen.com/track/pdf/10.1007/s41109-019-0189-1.pdf
 * transport ratio https://arxiv.org/ftp/arxiv/papers/1708/1708.00836.pdf

example outputs:

![cityengine streets on Chicago base; edge angle](https://github.com/twak/streetz/blob/master/examples/chicago_ce.npzEdge%20angle.png?raw=true)
![https://github.com/twak/streetz/blob/master/examples/all.png?raw=true](https://github.com/twak/streetz/blob/master/examples/all.png?raw=true)
