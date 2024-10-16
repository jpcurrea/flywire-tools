# flywire-tools
![flywire's 50 largest cells](./docs/fw_50_L.png)

## You _really_ should use the connectome.

### Flywire and Neuprint 
There are two established connectomes available with really user-friendly interfaces: [Flywire.ai](https://flywire.ai/) and [neuPrint+](https://neuprint.janelia.org/). I recommend you try them out to get a good idea of the kinds of data available. 

But the wiring diagram is HUGE (140,000 cells forming 50 million synapses). So if you plan to use them for analyzing connections between many different cells, you ought to work with the connectome programmatically. 

Here I've developed some tools for interacting with the flywire connectome. I've downloaded the spreadsheet containing all connections in the flywire connectome (with the latest available [here](https://codex.flywire.ai/api/download)). This allows for really large queries that are above the limit for the number of permitted calls for the flywire APIs.

Online interfaces also only provide connections with at least 5 synapses, by default, and a strict minimum of 2. Assuming these tools are only really useful for generating _hypotheses_ about cell and circuit function, I wanted a more statistical approach that queries all connections and allows for thresholding later in the pipeline.

Among these tools are routines for: 
  - Automated **pathway analysis** for connections going up- or downstream using a variety of search parameters
  - Measuring the **connectivity** between many different cell types and generating corresponding figures
  - Finding shared **downstream parters** of pairs of cell types, in order to identify cells likely to integrate those corresponding signals
  - Performing **Monte Carlo simulations** to empirically determine which pathways are more or less likely while accounting for nonlinearities in the wiring diagram
  - **Plotting** the results of these routines with systematic coloring schemes using neuroglancer

Much of this is still ongoing and all of it is has been built from the tools provided by the [navis module](https://navis-org.github.io/navis/) and its affiliates.


### Installation
The stable version of this library can be installed using pip:
```
pip install flywire-tools
```
Or, to download the current version straight from this repo:
```
pip install git+https://github.com/jpcurrea/flywire-tools.git
```
Alternatively, you can clone this repo and the run pip install from inside the cloned repo:
```
git clone https://github.com/jpcurrea/flywire-tools.git
cd flywire-tools
pip install .
```

## [Flywire through Navis: the basics](docs/startup.ipynb)

## [Documentation](docs/api_tutorial.ipynb)