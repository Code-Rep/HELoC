# HELoC: Hierarchical Contrastive Learning of Code Representations
A PyTorch Implementation of "HELoC: Hierarchical Contrastive Learning of Code Representations"
## Map Any Code Snippet into Vector Embedding with HELoC
HELoC is a self-supervised hierarchical contrastive learning model of code representation. Its key idea is to  formulate the learning of AST hierarchy as a pretext task of self-supervised contrastive learning, where cross-entropy and triplet losses are adopted as learning objectives to predict the level and learn the hierarchical relationships between nodes, which makes the representation vectors of nodes with greater differences in AST levels farther apart in the embedding space. By using such vectors, the structural similarities between code snippets can be measured more precisely. HELoC is self-supervised and can be applied to many source code related downstream tasks after pre-training. 
# Requirements <br />
pytorch 1.7.0 <br />
python 3.7.8 <br />
dgl 0.5.3 <br />
flair 0.7 <br />
pycparser 2.20 <br />
javalang 0.13.0 <br />
gensim 3.8.3 <br />
# Run <br />
## pre-training 
=======
# Usage
We extract the AST node embedding and path embedding in the following two steps:
1. run ```python parsercode.py --lang oj```/ ```python parsercode.py --lang gcj```/ ```python parsercode.py --lang bcb``` to generate initial encoding.
2. run ```python pre_training.py --dataset_nodeemb [The path to the dataset in which the nodes have been encoded]```
# Application of HELoC in downstream tasks
We evaluate HELoC model on two tasks, code classification and code clone detection. It is also expected to be helpful in more downstream tasks.
In the code classification task, we evaluate HELoC on two datasets: GCJ and OJ. In the code clone detection task, we further evaluate HELoC on three datasets: BCB, GCJ and OJClone. 
## Code Classification <br /> 
run ```python cla.py --nodedataset_path [node emb path] --pathdataset_path [path emb path] --pre_model [pre_model]```
## Code Clone Detection <br />
run ```python clo.py --dataset [The name of the dataset] --pair_file [The path of the clone pairs] --nodedataset_path [node emb path] --pathdataset_path [path emb path] --pre_model [pre_model]```
