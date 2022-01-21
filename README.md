# HELoC: Hierarchical Contrastive Learning of Source Code Representation
A PyTorch Implementation of "HELoC: Hierarchical Contrastive Learning of Source Code Representation"
## Map Any Code Snippet into Vector Embedding with HELoC
HELoC is a self-supervised hierarchical contrastive learning model of source code representation. Its key idea is to  formulate the learning of AST hierarchy as a pretext task of self-supervised contrastive learning, where node level prediction (NEP) and node relationship optimization (NRO) are adopted as learning objectives to predict the level and learn the three topological relationships between the nodes, which makes the representation vectors of nodes with greater differences in AST levels farther apart in the embedding space. This further manifests as the representation of the AST’s topology become more different in the embedding space, making it easier for the specific representation of a single AST to come to the surface and facilitating the downstream code similarity task.

In addition, we design a specialized GNN for AST hierarchical structures called RSGNN, which captures the AST hierarchy comprehensively by combining the self-attention mechanism ( trained for capturing global structure) with GCN ( skilled in capturing local structure). In addition, we add internal residual connection and external residual connection to the self-attention mechanism to address the gradient vanishing problem. 
HELoC is self-supervised and can be applied to many source code related downstream tasks after pre-training. 
# Requirements <br />
pytorch 1.7.0 <br />
python 3.7.8 <br />
dgl 0.5.3 <br />
flair 0.7 <br />
pycparser 2.20 <br />
javalang 0.13.0 <br />
gensim 3.8.3 <br />
# Pre-training Dataset <br />
The pre-training dataset we use is the Java-Large dataset published by [Code2vec](https://dl.acm.org/doi/abs/10.1145/3290353), which contains about 4 million Java files collected from GitHub.
# Run <br />
## pre-training 
# Usage
We extract the AST node embedding and path embedding in the following two steps:
```
run python pre_training.py\
--data_path [The path to the dataset] \
--device['cpu'/'cuda']
```
# Application of HELoC in downstream tasks
We evaluate HELoC model on three downstream tasks, code classification, code clone detection and code clustering. It is also expected to be helpful in more downstream tasks.
## Code Classification <br /> 
In the code classification task, we evaluate HELoC on two datasets: GCJ and OJ, where GCJ consists of java programs and OJ dataset is composed of C functions.
 ```
run python cla.py \
--data_path [The path to the dataset]\
--device ['cpu'/'cuda'] \
--pre_model [The path of the pre_model]
```
## Code Clone Detection <br />
In the code clone detection task, we further evaluate HELoC on three datasets: BCB, GCJ and OJClone. 

 ```
 run python clo.py \
--data_path [The path to the dataset] \
--device ['cpu'/'cuda']\
--pre_model [The path of the pre_model]\
--pair_file [The path of the clone pairs]
```
## Code Clustering <br />
In the code clustering task, we use the OJ and SA datasets for evaluation.

 ```
 run python clu.py \
 --data_path [The path to the dataset]\
--device ['cpu'/'cuda']\
--pre_model [The path of the pre_model]
```
# Compare to other work
Many existing methods aggregate nodes together based on parent-child connections to obtain the node representation so that the relationships among the adjacency levels can be captured. However, none of them are designed with the intention to learn the relationships between non-adjacent levels of AST nodes.
For example, methods (such as [ASTNN](https://ieeexplore.ieee.org/abstract/document/8812062), [TBCNN](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.740.9656&rep=rep1&type=pdf), and [HAG](https://www.worldscientific.com/doi/abs/10.1142/S021819402150025X)) learn the AST structure by aggregating child nodes to their parent node, focusing only on nodes adjacent hierarchy. Consequently, topologically nonadjacent hierarchy between nodes (sometimes spanning multiple levels) are difficult to learn, resulting in such relationships in the embedding space not being accurately determined. 
