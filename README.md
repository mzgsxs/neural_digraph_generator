<<<<<<< Updated upstream
# Neural di-graph generator
Generate directed graph (but possibly cyclic) by artifical neural networks 
=======
# Neural di-graph generator - a work in process 
Generate directed graph (but possibly cyclic) by artifical neural networks 
(part of this work was done while I was a PhD student at OIST) 
https://www.overleaf.com/read/mvbqpnqgtvkf  

Motivation: Many things in the world may viewed as graph, such as transpotation, biological brain, electronic circuits and social relations. This module was aimed to use in the following situation:  

" Given that I have a set of graphs (such as biological brain neural networks). I want to generate a new graph, but with similar graphical property as those existing ones. Thus, we learn the latent space of those graph by auto-encoder architechture, encoder is discarded and decoder is used for generating. Generated new graph is then used as the topology of an artificial neural network. A agent dapted with this artifical neural network is accesed in a particular enviroment. By generating many graphs(and agents) by perterbing the seed in the latent space, we can perform neuro-evolution and select the best performing graph. "  
>>>>>>> Stashed changes


to mimic the neural developing process,i.e genetic decoding and encoding into biological brain network

