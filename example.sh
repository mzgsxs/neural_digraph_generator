rm  ./exp/graph_auto_encoder/config*
rm  ./exp/graph_auto_encoder/events*
rm  ./exp/graph_auto_encoder/log*
# train model with configration digraph.yaml
python run_exp.py -c config/digraph_auto_encoder.yaml


